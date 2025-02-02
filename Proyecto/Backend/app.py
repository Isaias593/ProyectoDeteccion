from flask import Flask, request, jsonify, send_from_directory, render_template 
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
from datetime import datetime, timedelta
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from uuid import uuid4
import cv2
import base64
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import text
from DatabaseManager import db, Detection, DatabaseManager  # Asegúrate de importar correctamente
import json


# Configuración de Flask
app = Flask(
    __name__,
    static_folder='../Frontend',
    template_folder='../Frontend'
)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:12345nac@database-1.cvem2qsqaitl.us-east-1.rds.amazonaws.com:5432/database-1'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
#db = SQLAlchemy(app)

app.config['SQLALCHEMY_POOL_SIZE'] = 10
app.config['SQLALCHEMY_MAX_OVERFLOW'] = 5
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Inicializa SQLAlchemy
DatabaseManager.init_app(app) 



# Carpetas para archivos
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
DOWNLOAD_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# Carga del modelo YOLOv8
try:
    model = YOLO('modelo_entrenado.pt')  
    print("Modelo cargado exitosamente")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
CLASS_NAMES = model.names  # Mapea los índices de las clases a etiquetas

# Variables globales
live_streaming_active = False
live_detections = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/historial')
def historial():
    return render_template('historial.html')


@app.route('/test-connection', methods=['GET'])
def test_connection():
    try:
        # Ejecuta una consulta básica
        db.session.execute(text('SELECT 1'))
        return jsonify({'message': 'Conexion exitosa a la base de datos'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró el archivo'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó un archivo'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Verificar si es un video
    if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        return process_video(filepath, filename)

    # Procesar imágenes
    try:
        start_time = datetime.now()
        results = model(filepath)
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        detections = []
        for result in results:
            if not result.boxes:
                continue

            for box in result.boxes:
                bbox = str(box.xyxy[0].tolist())
                detection = {
                    'confidence': float(box.conf),
                    'label': CLASS_NAMES[int(box.cls)],
                    'processed': True,
                    'bbox': bbox
                }

                # Evitar duplicados en la base de datos
                existing_detection = Detection.query.filter_by(
                    filename=filename,
                    location=filepath,
                    vehicle_type=detection['label'],
                    bbox=bbox
                ).first()

                if existing_detection:
                    print("Detección duplicada detectada. No se guardará.")
                    continue

                # Guardar nueva detección
                new_detection = Detection(
                    filename=filename,
                    location=filepath,
                    vehicle_type=detection['label'],
                    confidence=detection['confidence'],
                    timestamp=datetime.now(),
                    processing_time=processing_time,
                    processed=True,
                    bbox=bbox
                )
                db.session.add(new_detection)
                db.session.flush()
                detection['id'] = new_detection.id
                db.session.commit()

                detections.append(detection)

        # Guardar la imagen procesada
        processed_image_path = os.path.join(PROCESSED_FOLDER, filename)
        annotated_image = results[0].plot()
        cv2.imwrite(processed_image_path, annotated_image)

        response = {
            'filename': filename,
            'location': filepath,
            'analysis_date': datetime.now().isoformat(),
            'total_objects': len(detections),
            'detections': detections,
            'processing_time': processing_time,
            'processed_image': f'/processed/{filename}'
        }

        try:
            socketio.emit('progress', {'message': 'Procesamiento completado', 'data': response})
        except Exception as e:
            print(f"Error emitiendo evento de progreso: {e}")

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def calculate_iou(box1, box2):
    """Calcula el IoU (Intersection over Union) entre dos bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    # Coordenadas del área de intersección
    inter_x1 = max(x1, x1_)
    inter_y1 = max(y1, y1_)
    inter_x2 = min(x2, x2_)
    inter_y2 = min(y2, y2_)

    # Área de intersección
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Áreas de los rectángulos individuales
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)

    # IoU
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def filter_duplicates(new_detections, previous_detections, iou_threshold=0.5, ttl=5):
    current_time = datetime.now()
    previous_detections = [
        det for det in previous_detections
        if (current_time - det['timestamp']).seconds < ttl
    ]
    filtered = []
    for new_det in new_detections:
        is_duplicate = False
        for prev_det in previous_detections:
            if calculate_iou(new_det['bbox'], prev_det['bbox']) > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            filtered.append(new_det)
    return filtered


def process_video(filepath, filename):
    cap = cv2.VideoCapture(filepath)
    frame_skip = 3
    frame_count = 0
    previous_detections = []
    start_time = datetime.now()  # Inicio del procesamiento del video

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            frame_count += 1
            results = model(frame)
            annotated_frame = results[0].plot()

            current_detections = []
            for result in results:
                for box in result.boxes:
                    bbox = list(map(int, box.xyxy[0].tolist()))
                    current_detections.append({
                        'confidence': float(box.conf),
                        'label': CLASS_NAMES[int(box.cls)],
                        'bbox': bbox,
                        'timestamp': datetime.now()
                    })

            # Filtrar duplicados en memoria
            unique_detections = filter_duplicates(current_detections, previous_detections)
            previous_detections.extend(unique_detections)

            # Calcular el tiempo de procesamiento acumulado
            processing_time = (datetime.now() - start_time).total_seconds()

            # Guardar en la base de datos
            for det in unique_detections:
                try:
                    save_detection(
                        filename=filename,
                        location=filepath,
                        vehicle_type=det['label'],
                        confidence=det['confidence'],
                        timestamp=det['timestamp'],
                        processing_time=processing_time,  # Pasamos el tiempo de procesamiento aquí
                        processed=True,
                        bbox=det['bbox']
                    )
                except Exception as e:
                    print(f"Error guardando detección de video: {e}")

            # Emitir detecciones al frontend
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_data = base64.b64encode(buffer).decode('utf-8')

            socketio.emit('video_frame', {
                'frame': frame_data,
                'detections': [
                    {
                        'confidence': det['confidence'],
                        'label': det['label'],
                        'bbox': det['bbox'],
                        'timestamp': det['timestamp'].isoformat()
                    }
                    for det in unique_detections
                ],
                'filename': filename,
                'location': filepath,  # Ruta del archivo
                'processing_time': processing_time  # Incluir el tiempo de procesamiento
            })

        cap.release()
        socketio.emit('video_completed', {'message': f'Procesamiento del video \"{filename}\" completado.'})
        return jsonify({'message': 'Procesando video...'}), 200

    except Exception as e:
        cap.release()
        return jsonify({'error': str(e)}), 500


def save_detection(filename, location, vehicle_type, confidence, timestamp, processing_time, processed=True, bbox=None):
    """
    Guarda una detección en la base de datos si no es duplicada según IoU y tiempo.
    """
    # Configurar umbrales
    iou_threshold = 0.5
    time_threshold = 1.0  # En segundos

    try:
        # Validar que bbox esté presente y en el formato correcto
        if not bbox or not isinstance(bbox, list) or len(bbox) != 4:
            print("Bounding box inválido. Detección no guardada.")
            return

        # Filtrar detecciones existentes en un rango de tiempo
        time_window_start = timestamp - timedelta(seconds=time_threshold)
        time_window_end = timestamp + timedelta(seconds=time_threshold)

        existing_detections = Detection.query.filter(
            Detection.filename == filename,
            Detection.location == location,
            Detection.vehicle_type == vehicle_type,
            Detection.timestamp.between(time_window_start, time_window_end)
        ).all()

        for existing_detection in existing_detections:
            # Parsear el bounding box existente
            try:
                existing_bbox = [float(coord) for coord in existing_detection.bbox.strip('[]').split(',')]
            except ValueError:
                print(f"BBox inválido en detección existente (ID {existing_detection.id}). Saltando.")
                continue

            # Calcular IoU
            if calculate_iou(bbox, existing_bbox) > iou_threshold:
                print("Detección duplicada detectada. No se guardará.")
                return

        # Guardar la nueva detección si no es duplicada
        new_detection = Detection(
            filename=filename,
            location=location,
            vehicle_type=vehicle_type,
            confidence=confidence,
            timestamp=timestamp,
            processing_time=processing_time,
            processed=processed,
            bbox=str(bbox) if bbox else None
        )
        db.session.add(new_detection)
        db.session.commit()
        print("Nueva detección guardada.")

    except Exception as e:
        print(f"Error al guardar la detección: {e}")


#busccar por fecha
@app.route('/buscar-por-fecha', methods=['GET'])
def buscar_por_fecha():
    """
    Endpoint para buscar detecciones por fecha específica.
    """
    fecha = request.args.get('fecha')  # Obtener la fecha de los parámetros de la URL
    if not fecha:
        return jsonify({"error": "Se requiere el parámetro 'fecha' en formato YYYY-MM-DD"}), 400

    # Validar el formato de la fecha
    try:
        datetime.strptime(fecha, '%Y-%m-%d')
    except ValueError:
        return jsonify({"error": "El formato de la fecha debe ser YYYY-MM-DD"}), 400

    try:
        # Consultar la base de datos con SQLAlchemy para filtrar detecciones por fecha
        detecciones = Detection.query.filter(
            db.func.date(Detection.timestamp) == fecha,
            Detection.processed == True
        ).all()

        if not detecciones:
            return jsonify({"message": "No se encontraron registros para la fecha seleccionada"}), 200

        # Convertir los resultados a formato JSON
        resultado = [
            {
                "id": d.id,
                "filename": d.filename,
                "location": d.location,
                "vehicle_type": d.vehicle_type,
                "confidence": d.confidence,
                "timestamp": d.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "processing_time": d.processing_time,
                "processed": 'Sí' if d.processed else 'No',
                "bbox": json.loads(d.bbox) if d.bbox else None  # Convertir bbox a JSON si está presente
            }
            for d in detecciones
        ]

        return jsonify(resultado), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


#fin

@app.route('/generate_report_by_date', methods=['POST'])
def generate_report_by_date():
    data = request.get_json()
    fecha = data.get('fecha')  # "YYYY-MM-DD"

    if not fecha:
        return jsonify({'error': 'No se proporcionó fecha'}), 400

    # 1. Buscar las detecciones en la BD según esa fecha
    detecciones = Detection.query.filter(
        db.func.date(Detection.timestamp) == fecha
    ).all()

    # 2. Convertir a lista de dict
    detections_list = []
    for det in detecciones:
        detections_list.append({
            'id': det.id,
            'filename': det.filename,
            'location': det.location,
            'vehicle_type': det.vehicle_type,
            'confidence': det.confidence,
            'timestamp': det.timestamp.isoformat(),
            'processing_time': det.processing_time,
        })

    # 3. Generar el archivo txt igual que tu /generate_report
    #    (o llama a una función auxiliar que haga la lógica de escribir archivo)
    report_name = secure_filename(f'report_{uuid4().hex}.txt')
    report_path = os.path.join(DOWNLOAD_FOLDER, report_name)
    
    try:
        with open(report_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write("Reporte de Detección de Vehículos\n\n")
            for d in detections_list:
                txt_file.write(
                    f"ID: {d['id']}, "
                    f"Nombre: {d['filename']}, "
                    f"Ubicación: {d['location']}, "
                    f"Tipo: {d['vehicle_type']}, "
                    f"Confianza: {d['confidence']:.2f}%, "
                    f"Fecha/Hora: {d['timestamp']}, "
                    f"Tiempo: {d['processing_time']}s\n"
                )
    except Exception as e:
        return jsonify({'error': f'Error al escribir el reporte: {str(e)}'}), 500

    return jsonify({
        'message': 'Reporte generado correctamente.',
        'report_path': f'/download/{report_name}'
    }), 200

@app.route('/processed/<filename>')
def processed_file(filename):
    """
    Endpoint para servir archivos procesados al cliente.
    """
    try:
        # Asegurar que el nombre del archivo es seguro
        safe_filename = secure_filename(filename)
        file_path = os.path.join(PROCESSED_FOLDER, safe_filename)

        # Verificar si el archivo existe en la carpeta especificada
        if not os.path.exists(file_path):
            return jsonify({'error': f'El archivo "{safe_filename}" no existe en la carpeta procesada.'}), 404

        # Enviar el archivo desde el directorio seguro
        return send_from_directory(PROCESSED_FOLDER, safe_filename, as_attachment=True)

    except Exception as e:
        # Manejar cualquier error inesperado
        return jsonify({'error': f'Error al intentar enviar el archivo: {str(e)}'}), 500


@app.route('/detections', methods=['GET'])
def get_detections():
    """
    Endpoint para obtener las detecciones con soporte de paginación y filtros opcionales de fecha.
    """
    try:
        # Parámetros de consulta opcionales
        page = int(request.args.get('page', 1))  # Página actual, predeterminada en 1
        per_page = int(request.args.get('per_page', 10))  # Elementos por página, predeterminada en 10
        start_date = request.args.get('start_date')  # Filtro de fecha inicial (YYYY-MM-DD)
        end_date = request.args.get('end_date')  # Filtro de fecha final (YYYY-MM-DD)

        # Construir consulta base
        query = Detection.query

        # Filtrar por fecha inicial
        if start_date:
            try:
                start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
                query = query.filter(Detection.timestamp >= start_date_obj)
            except ValueError:
                return jsonify({'error': 'Formato de fecha inválido para start_date. Use YYYY-MM-DD.'}), 400

        # Filtrar por fecha final
        if end_date:
            try:
                end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
                query = query.filter(Detection.timestamp <= end_date_obj)
            except ValueError:
                return jsonify({'error': 'Formato de fecha inválido para end_date. Use YYYY-MM-DD.'}), 400

        # Ordenar por fecha descendente y aplicar paginación
        detections_paginated = query.order_by(Detection.timestamp.desc()).paginate(page, per_page, False)

        # Convertir detecciones en formato JSON
        result = [
            {
                'id': d.id,
                'filename': d.filename or 'N/A',
                'location': d.location or 'N/A',
                'vehicle_type': d.vehicle_type or 'Desconocido',
                'confidence': round(d.confidence * 100, 2),  # Confianza en porcentaje
                'timestamp': d.timestamp.isoformat(),  # Fecha/hora en formato ISO
                'processing_time': d.processing_time,
                'processed': 'Sí' if d.processed else 'No',
                'bbox': json.loads(d.bbox) if d.bbox else None  # Convertir bbox a lista si está presente
            }
            for d in detections_paginated.items
        ]

        # Construir respuesta con datos de paginación
        response = {
            'page': detections_paginated.page,
            'per_page': detections_paginated.per_page,
            'total': detections_paginated.total,
            'detections': result
        }
        return jsonify(response), 200

    except Exception as e:
        # Manejo de errores
        return jsonify({'error': f'Error al obtener detecciones: {str(e)}'}), 500



@app.route('/clear_detections', methods=['DELETE'])
def clear_detections():
    """
    Elimina registros de detecciones, opcionalmente filtrados por fecha.
    """
    fecha = request.args.get('fecha')  # Parámetro opcional de fecha
    confirm = request.args.get('confirm', 'false').lower()  # Confirmación explícita requerida

    # Validar confirmación explícita
    if confirm != 'true':
        return jsonify({'error': 'Confirma la acción con el parámetro confirm=true.'}), 400

    try:
        # Crear consulta base
        query = db.session.query(Detection)

        # Validar y filtrar por fecha si se proporciona
        if fecha:
            try:
                fecha_obj = datetime.strptime(fecha, '%Y-%m-%d').date()
                query = query.filter(db.func.date(Detection.timestamp) == fecha_obj)
            except ValueError:
                return jsonify({'error': 'El formato de la fecha debe ser YYYY-MM-DD.'}), 400

        # Eliminar registros y obtener conteo
        count = query.delete()
        db.session.commit()

        # Reiniciar la secuencia solo si no hay registros en la tabla
        if db.session.query(Detection).count() == 0:
            db.session.execute("ALTER SEQUENCE detections_id_seq RESTART WITH 1")
            db.session.commit()

        # Construir respuesta
        if count == 0:
            return jsonify({'message': 'No se encontraron registros para eliminar.'}), 200

        message = f'Se eliminaron {count} registro(s).'
        if fecha:
            message += f' Filtrados por la fecha: {fecha}.'
        return jsonify({'message': message}), 200

    except Exception as e:
        # Manejar errores y deshacer cambios en la base de datos
        db.session.rollback()
        return jsonify({'error': f'Error al intentar eliminar registros: {str(e)}'}), 500


@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        data = request.get_json()
        detections = data.get('detections')

        # Validar que detections sea una lista válida y no esté vacía
        if not isinstance(detections, list) or not detections:
            return jsonify({'error': 'La lista de detecciones es inválida o está vacía.'}), 400

        # Validar que cada detección tenga campos obligatorios
        required_fields = ['id', 'filename', 'location', 'vehicle_type', 'confidence', 'timestamp', 'processing_time']
        for i, detection in enumerate(detections, start=1):
            if not all(field in detection for field in required_fields):
                return jsonify({
                    'error': f'Faltan campos obligatorios en la detección #{i}: {detection}'
                }), 400

        # Generar nombre único para el reporte
        report_name = secure_filename(f'report_{uuid4().hex}.txt')
        report_path = os.path.join(DOWNLOAD_FOLDER, report_name)

        try:
            with open(report_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write("Reporte de Detección de Vehículos\n\n")

                for detection in detections:
                    # -----------------------------------
                    # Limpia el símbolo '%' de la confianza,
                    # convirtiéndola a número
                    conf_str = str(detection['confidence'])
                    conf_str = conf_str.replace('%', '').strip()  # ej: "95.50%" -> "95.50"
                    try:
                        conf_val = float(conf_str)
                    except ValueError:
                        conf_val = 0.0  # o podrías lanzar excepción si prefieres

                    txt_file.write(
                        f"ID: {detection['id']}, "
                        f"Nombre: {detection['filename']}, "
                        f"Ubicación: {detection['location']}, "
                        f"Tipo: {detection['vehicle_type']}, "
                        f"Confianza: {conf_val:.2f}%, "  # Se muestra con 2 decimales
                        f"Fecha/Hora: {detection['timestamp']}, "
                        f"Tiempo: {detection['processing_time']}s\n"
                    )
        except Exception as e:
            return jsonify({'error': f'Error al escribir el reporte: {str(e)}'}), 500

        # Responder con la ruta del reporte
        return jsonify({'message': 'Reporte generado correctamente.', 'report_path': f'/download/{report_name}'}), 200

    except Exception as e:
        return jsonify({'error': f'Error al generar el reporte: {str(e)}'}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    try:
        # Asegurar que el nombre del archivo es seguro
        safe_filename = secure_filename(filename)
        file_path = os.path.join(DOWNLOAD_FOLDER, safe_filename)

        # Verificar si el archivo existe
        if not os.path.exists(file_path):
            return jsonify({'error': f'El archivo "{safe_filename}" no se encontró en el servidor.'}), 404

        # Verificar si el archivo tiene permisos de lectura
        if not os.access(file_path, os.R_OK):
            return jsonify({'error': f'No se puede leer el archivo "{safe_filename}". Verifique los permisos.'}), 403

        # Enviar el archivo al cliente
        return send_from_directory(DOWNLOAD_FOLDER, safe_filename, as_attachment=True)

    except Exception as e:
        return jsonify({'error': f'Error al intentar descargar el archivo "{filename}": {str(e)}'}), 500


live_streaming_active = False
detected_objects = []  # Lista para almacenar objetos detectados
next_id = 1  # Contador para asignar IDs únicos
iou_threshold = 0.5  # Umbral para determinar si un objeto es el mismo
ttl_seconds = 5  # Tiempo de vida de los objetos no vistos


def calculate_iou(box1, box2):
    """Calcula el IoU (Intersection over Union) entre dos bounding boxes."""
    if not box1 or not box2 or len(box1) != 4 or len(box2) != 4:
        return 0  # Devuelve 0 si las cajas no son válidas

    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    # Coordenadas del área de intersección
    inter_x1 = max(x1, x1_)
    inter_y1 = max(y1, y1_)
    inter_x2 = min(x2, x2_)
    inter_y2 = min(y2, y2_)

    # Área de intersección
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Áreas de los rectángulos individuales
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)

    # IoU
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def clean_up_old_objects():
    """Limpia objetos no vistos dentro del TTL definido."""
    global detected_objects
    now = datetime.now()
    detected_objects = [
        obj for obj in detected_objects if (now - obj['last_seen']).seconds <= ttl_seconds
    ]


def track_objects(current_detections):
    """Rastrea objetos detectados y les asigna IDs únicos."""
    global next_id, detected_objects
    tracked_objects = []

    for detection in current_detections:
        matched = False
        for obj in detected_objects:
            iou = calculate_iou(detection['bbox'], obj['bbox'])
            if iou > iou_threshold:
                # Actualizar objeto rastreado
                obj['bbox'] = detection['bbox']
                obj['last_seen'] = datetime.now()
                obj['confidence'] = detection['confidence']
                tracked_objects.append(obj)
                matched = True
                break

        if not matched:
            # Asignar nuevo ID a objetos no coincidentes
            detection['id'] = next_id
            detection['new'] = True  # Marcar como nuevo
            next_id += 1
            detection['last_seen'] = datetime.now()
            tracked_objects.append(detection)

    # Actualizar lista global de objetos rastreados
    detected_objects = tracked_objects
    return tracked_objects


def clean_up_objects():
    """Elimina objetos que no se han visto recientemente."""
    global detected_objects
    now = datetime.now()
    detected_objects = [
        obj for obj in detected_objects if (now - obj['last_seen']).seconds <= ttl_seconds
    ]




@socketio.on('start_streaming')
def start_streaming():
    global live_streaming_active
    live_streaming_active = True
    cap = cv2.VideoCapture(0)  # reemplazar camara
    frame_skip = 10
    frame_count = 0

    try:
        while live_streaming_active:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            frame_count += 1

            results = model(frame)
            annotated_frame = results[0].plot()

            # Extraer detecciones "crudas"
            detections_raw = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    detections_raw.append({
                        'confidence': float(box.conf),
                        'label': CLASS_NAMES[int(box.cls)],
                        'bbox': [x1, y1, x2, y2]
                    })

            # Tracking básico
            tracked = track_objects(detections_raw)
            clean_up_objects()

            # Guardar en BD (solo objetos "nuevos")
            for obj in tracked:
                if obj.get('new'):
                    try:
                        # Convertir datetime a string para la BD
                        current_time = datetime.now()
                        save_detection(
                            filename='LiveStream',
                            location='Camara0',
                            vehicle_type=obj['label'],
                            confidence=obj['confidence'],
                            timestamp=current_time,  # Se maneja en save_detection
                            processing_time=0.0,
                            processed=True,
                            bbox=obj['bbox']
                        )
                    except Exception as e:
                        print(f"Error guardando detección de streaming: {e}")

            # Convertir 'last_seen' a string para no causar error JSON
            for obj in tracked:
                if 'last_seen' in obj and isinstance(obj['last_seen'], datetime):
                    obj['last_seen'] = obj['last_seen'].isoformat()

            # Emitir frame anotado
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_data = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('frame', {
                'frame': frame_data,
                'detections': tracked  # 'tracked' ya no tiene datetime
            })

        cap.release()
        live_streaming_active = False
        socketio.emit('stream_stopped', {'message': 'El streaming se detuvo correctamente.'})

    except Exception as e:
        cap.release()
        print(f"Error en el streaming: {e}")
        socketio.emit('stream_error', {'error': str(e)})


def track_objects(current_detections):
    """
    Asigna IDs y mantiene un registro de 'last_seen' como string.
    """
    global next_id, detected_objects
    tracked_objects = []

    for detection in current_detections:
        matched = False
        for obj in detected_objects:
            iou = calculate_iou(detection['bbox'], obj['bbox'])
            if iou > iou_threshold:
                # Actualizar
                obj['bbox'] = detection['bbox']
                obj['confidence'] = detection['confidence']
                # Convierto last_seen a string para evitar error en socketio.emit
                obj['last_seen'] = datetime.now().isoformat()
                tracked_objects.append(obj)
                matched = True
                break

        if not matched:
            detection['id'] = next_id
            detection['new'] = True
            next_id += 1
            # last_seen en string
            detection['last_seen'] = datetime.now().isoformat()
            tracked_objects.append(detection)

    # Actualizar la lista global
    detected_objects = tracked_objects
    return tracked_objects


def clean_up_objects():
    """Elimina objetos no vistos recientemente (ttl_seconds)."""
    global detected_objects
    now = datetime.now()
    detected_objects = [
        obj for obj in detected_objects
        if (now - datetime.fromisoformat(obj['last_seen'])).seconds <= ttl_seconds
    ]


@socketio.on('stop_stream')
def stop_stream():
    global live_streaming_active, detected_objects, next_id
    try:
        if live_streaming_active:
            # Detener el streaming en vivo
            live_streaming_active = False
            detected_objects.clear()  # Limpia la lista de objetos detectados
            next_id = 1  # Reinicia el contador de IDs
            print("Streaming detenido por cliente.")
            
            # Emitir evento al cliente para confirmar que se ha detenido
            socketio.emit('stream_stopped', {'message': 'El streaming en vivo ha sido detenido.'})

        else:
            print("El streaming ya estaba detenido.")
            socketio.emit('stream_stopped', {'message': 'El streaming ya estaba detenido.'})

    except Exception as e:
        # En caso de error, manejar la excepción
        print(f"Error al detener el streaming: {e}")
        socketio.emit('stream_error', {'error': str(e)})




if __name__ == '__main__':
    # Crear tablas en la base de datos
    DatabaseManager.crear_tablas(app)
    socketio.run(app, host="0.0.0.0", port=5000)
