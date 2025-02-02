// ============================
        // 1. Variables Globales
        // ============================
        const socket = io();             // Asumiendo que ya tienes socket.io disponible
        let isStreaming = false;
        let currentIdCounter = 1;
        let detectedObjects = [];        // Para el control de duplicados (streaming)
        const detectionTimeout = 3000;   // 3 segundos para limpiar detecciones antiguas

        // ============================
        // 2. Función de Previsualización
        // ============================
        function previewImage(event) {
            const file = event.target.files[0];
            if (!file) {
                alert("Por favor selecciona un archivo.");
                return;
            }

            // Tipos de archivos permitidos
            const allowedTypes = ['image/jpeg', 'image/png', 'image/jpg', 'video/mp4', 'video/avi', 'video/mkv'];
            if (!allowedTypes.includes(file.type)) {
                alert("Solo se permiten imágenes (JPEG, PNG) o videos (MP4, AVI, MKV).");
                return;
            }

            // Validar tamaño del archivo (opcional)
            const maxFileSize = 50 * 1024 * 1024; // Tamaño máximo: 50 MB
            if (file.size > maxFileSize) {
                alert("El archivo excede el tamaño máximo permitido (50 MB).");
                return;
            }

            const reader = new FileReader();
            const loadingSpinner = document.getElementById("loadingSpinner");

            reader.onloadstart = function () {
                loadingSpinner.style.display = "block";
            };

            reader.onloadend = function () {
                loadingSpinner.style.display = "none";
            };

            reader.onload = function (e) {
                const preview = document.getElementById('imagePreview');
                const videoPreviewText = document.getElementById("videoPreviewText");

                if (file.type.startsWith("video")) {
                    // Si es video, mostramos texto en lugar de la imagen
                    preview.style.display = "none";
                    videoPreviewText.textContent = "Archivo de video cargado. Será procesado.";
                    videoPreviewText.style.display = "block";
                } else {
                    // Si es imagen
                    videoPreviewText.style.display = "none";
                    preview.src = e.target.result;
                    preview.style.display = "block";
                }
            };

            reader.readAsDataURL(file);
        }

        // ============================
        // 3. Utilidades: IoU y Limpieza de Detecciones
        // ============================
        /**
         * Calcula el IoU (Intersection over Union) entre dos bounding boxes.
         * @param {Array} box1 - [x1, y1, x2, y2].
         * @param {Array} box2 - [x1, y1, x2, y2].
         * @returns {number} - Valor del IoU.
         */
        function calculateIoU(box1, box2) {
            const [x1, y1, x2, y2] = box1;
            const [x1_, y1_, x2_, y2_] = box2;

            const interX1 = Math.max(x1, x1_);
            const interY1 = Math.max(y1, y1_);
            const interX2 = Math.min(x2, x2_);
            const interY2 = Math.min(y2, y2_);

            const interArea = Math.max(0, interX2 - interX1) * Math.max(0, interY2 - interY1);
            const box1Area = (x2 - x1) * (y2 - y1);
            const box2Area = (x2_ - x1_) * (y2_ - y1_);
            const unionArea = box1Area + box2Area - interArea;

            if (unionArea === 0) return 0;
            return interArea / unionArea;
        }

        /**
         * Verifica si un objeto ya ha sido detectado en un período reciente.
         * @param {Array} detectedObjects - Lista de detecciones ya registradas.
         * @param {Array} newBox - Bounding box de la nueva detección.
         * @param {number} currentTimestamp - Marca de tiempo actual.
         * @param {number} timeout - Ventana de tiempo para considerar una detección como duplicada.
         * @returns {boolean} - `true` si la detección es duplicada, `false` en caso contrario.
         */
        function isDuplicateDetection(detectedObjects, newBox, currentTimestamp, timeout) {
            const iouThreshold = 0.5;
            return detectedObjects.some(obj => {
                // Verificar que no haya pasado el tiempo de 'timeout'
                const timeDiff = currentTimestamp - obj.timestamp;
                if (timeDiff > timeout) return false;

                // Calcular IoU
                const iou = calculateIoU(obj.bbox, newBox);
                return iou >= iouThreshold;
            });
        }

        /**
         * Limpia detecciones antiguas de la lista (basado en `detectionTimeout`).
         * @param {Array} detectedObjects - Lista global de detecciones.
         * @param {number} currentTimestamp - Marca de tiempo actual.
         * @param {number} timeout - Tiempo límite para considerar detecciones antiguas.
         */
        function cleanUpDetectedObjects(detectedObjects, currentTimestamp, timeout) {
            const beforeCleanup = detectedObjects.length;
            // Mantener solo las que estén dentro del rango de tiempo
            const filtered = detectedObjects.filter(obj => {
                return (currentTimestamp - obj.timestamp) <= timeout;
            });
            const removed = beforeCleanup - filtered.length;
            if (removed > 0) {
                console.log(`${removed} detección(es) antigua(s) eliminada(s).`);
            }
            // Actualizar la lista original
            detectedObjects.length = 0;
            detectedObjects.push(...filtered);
        }

        // ============================
        // 4. Funciones de Formato
        // ============================
        function formatBBox(bbox) {
            try {
                // bbox puede llegar como string JSON o como array
                const arr = (typeof bbox === 'string') ? JSON.parse(bbox) : bbox;
                if (!Array.isArray(arr) || arr.length < 4) return 'N/A';
                const [x1, y1, x2, y2] = arr;
                return `(${x1.toFixed(2)}, ${y1.toFixed(2)}) - (${x2.toFixed(2)}, ${y2.toFixed(2)})`;
            } catch {
                return 'N/A';
            }
        }

        function formatDate(timestamp) {
            try {
                const date = new Date(timestamp);
                return date.toLocaleDateString('es-ES') + ' ' + date.toLocaleTimeString('es-ES');
            } catch {
                return 'Fecha inválida';
            }
        }

        // ============================
        // 5. Función para Agregar Detecciones a la Tabla Actual
        // ============================
        let currentTempIdCounter = 1; // Contador temporal de IDs que comienza desde 1

function addToCurrentTable(detection) {
    const table = document.getElementById('currentDetectionsTable');
    const totalObjects = document.getElementById('totalObjectsCurrent');

    // **Usar un ID temporal para la tabla de detecciones actuales**
    const id = currentTempIdCounter++; // Incrementar el contador de IDs temporales

    // Otros valores con sus predeterminados
    const filename = detection.filename || "Sin archivo";
    const location = detection.location || "Desconocido";
    const vehicleType = detection.vehicle_type || "Desconocido";
    const confidence = detection.confidence ? `${(detection.confidence * 100).toFixed(2)}%` : "N/A";
    const timestamp = detection.timestamp ? formatDate(detection.timestamp) : "Fecha desconocida";
    const processingTime = (detection.processing_time && !isNaN(detection.processing_time))
        ? `${Number(detection.processing_time).toFixed(2)} s`
        : "N/A";
    const processed = detection.processed ? "Sí" : "No";
    const bbox = formatBBox(detection.bbox);

    // Crear fila
    const row = document.createElement('tr');
    row.innerHTML = `
        <td>${id}</td> <!-- Usar ID temporal -->
        <td>${filename}</td>
        <td>${location}</td>
        <td>${vehicleType}</td>
        <td>${confidence}</td>
        <td>${timestamp}</td>
        <td>${processingTime}</td>
        <td>${processed}</td>
        <td>${bbox}</td>
    `;
    table.appendChild(row);

    // Actualizar contador de objetos totales
    const currentTotal = parseInt(totalObjects.textContent) || 0;
    totalObjects.textContent = currentTotal + 1;
}

// Reiniciar el contador temporal al iniciar nuevas detecciones
function resetCurrentTempIdCounter() {
    currentTempIdCounter = 1; // Reiniciar el contador a 1
}

        // ============================
        // 6. Manejador del Formulario de Subida
        // ============================
        document.getElementById('uploadForm').addEventListener('submit', async function (event) {
            event.preventDefault();
            const formData = new FormData(event.target);

            // Mostrar spinner
            const spinner = document.getElementById('loadingSpinner');
            spinner.style.display = 'block';

            try {
                alert("Cargando y procesando el archivo. Esto puede tardar un momento...");

                const response = await axios.post('/upload', formData, {
                    headers: { 'Content-Type': 'multipart/form-data' },
                });

                const {
                    detections,
                    processed_image,
                    filename,
                    location,
                    analysis_date,
                    processing_time
                } = response.data;

                if (!detections || detections.length === 0) {
                    alert('El archivo fue procesado, pero no se encontraron detecciones.');
                    return;
                }

                // Reiniciar ID local y la lista de objetos detectados
                currentIdCounter = 1;
                detectedObjects.length = 0;

                // Agregar cada detección a la tabla (evitando duplicados si fuera necesario)
                const now = Date.now();
                detections.forEach(d => {
                    if (!d.bbox) return;

                    // Verificamos si es duplicado por IoU
                    if (isDuplicateDetection(detectedObjects, d.bbox, now, detectionTimeout)) {
                        // Ya detectado recientemente
                        return;
                    }

                    // Agregamos al arreglo global
                    detectedObjects.push({
                        bbox: (typeof d.bbox === 'string') ? JSON.parse(d.bbox) : d.bbox,
                        timestamp: now
                    });

                    // Agregamos a la tabla
                    addToCurrentTable({
                        id: d.id || null,
                        filename: filename || 'Archivo',
                        location: location || 'Desconocido',
                        vehicle_type: d.label,
                        confidence: d.confidence,
                        bbox: d.bbox,
                        timestamp: analysis_date || now,
                        processing_time: processing_time || 0,
                        processed: true
                    });
                });

                // Mostrar imagen procesada (si viene)
                if (processed_image) {
                    const resultImage = document.getElementById('imagePreview');
                    resultImage.src = processed_image;
                    resultImage.style.display = 'block';
                }

                alert('Archivo procesado correctamente.');

            } catch (error) {
                console.error('Error al cargar el archivo:', error);
                const errorMessage = error.response?.data?.error || error.message || 'Error desconocido';
                alert(`Error al cargar el archivo: ${errorMessage}`);
            } finally {
                spinner.style.display = 'none';
            }
        });

        // ============================
        // 7. Streaming (Start / Stop)
        // ============================
        document.getElementById('startStream').addEventListener('click', () => {
            if (!isStreaming) {
                isStreaming = true;
                socket.emit('start_streaming');
                console.log("Streaming iniciado");

                // Limpiar tabla y contador
                document.getElementById('currentDetectionsTable').innerHTML = '';
                document.getElementById('totalObjectsCurrent').textContent = '0';

                // Limpiar la lista local de detecciones
                detectedObjects.length = 0;

                // Suscribirse al evento 'frame' (llegan las imágenes en vivo)
                socket.on('frame', (data) => {
                    // Mostrar imagen en vivo
                    const liveFrame = document.getElementById('liveFrame');
                    liveFrame.src = 'data:image/jpeg;base64,' + data.frame;

                    // Procesar detecciones en tiempo real
                    if (data.detections) {
                        const now = Date.now();
                        data.detections.forEach(detection => {
                            const bbox = detection.bbox || [0,0,0,0];
                            // Verificar duplicado
                            if (!isDuplicateDetection(detectedObjects, bbox, now, detectionTimeout)) {
                                // Agregamos a la tabla y a la lista local
                                detectedObjects.push({
                                    bbox,
                                    timestamp: now
                                });
                                addToCurrentTable({
                                    id: detection.id || `Live-${currentIdCounter++}`,
                                    filename: 'Streaming',
                                    location: 'En vivo',
                                    vehicle_type: detection.label,
                                    confidence: detection.confidence,
                                    bbox: bbox,
                                    timestamp: now,
                                    processing_time: 0,
                                    processed: true
                                });
                            }
                        });
                        // Limpiar detecciones viejas
                        cleanUpDetectedObjects(detectedObjects, now, detectionTimeout);
                    }
                });
            }
        });

        document.getElementById('stopStream').addEventListener('click', () => {
            if (isStreaming) {
                socket.emit('stop_stream');
                console.log("Evento 'stop_stream' emitido al servidor");
                isStreaming = false;

                // Remover listener para no acumularlo en cada start
                socket.off('frame');

                // Limpiar la vista previa del streaming
                document.getElementById('liveFrame').src = '';
            }
        });

        // ============================
        // 8. Evento para "video_frame" (procesamiento en backend, si aplica)
        //    - Solo si tu backend emite este evento con frames procesados
        // ============================
        socket.on('video_frame', (data) => {
            const videoFrame = document.getElementById('videoFrame');
            videoFrame.src = 'data:image/jpeg;base64,' + data.frame;
        
            // Asegurarse de que hay detecciones procesadas
            if (data.detections) {
                const now = Date.now();
                data.detections.forEach(d => {
                    // Evitar duplicados
                    if (!isDuplicateDetection(detectedObjects, d.bbox, now, detectionTimeout)) {
                        detectedObjects.push({ bbox: d.bbox, timestamp: now });
        
                        // Asignar valores desde el backend
                        const filename = data.filename || 'Sin nombre'; // Nombre del archivo subido
                        const location = data.location || 'Sin ubicación'; // Ruta del archivo
                        const processingTime = data.processing_time || 'N/A'; // Tiempo de procesamiento
        
                        // Agregar a la tabla de detecciones actuales
                        addToCurrentTable({
                            id: d.id || `VF-${currentIdCounter++}`,
                            filename: filename, // Mostrar el nombre del archivo real
                            location: location, // Mostrar la ruta del archivo subido
                            vehicle_type: d.label,
                            confidence: d.confidence,
                            bbox: d.bbox,
                            timestamp: now,
                            processing_time: processingTime, // Mostrar el tiempo de procesamiento real
                            processed: true
                        });
                    }
                });
        
                // Limpiar detecciones obsoletas
                cleanUpDetectedObjects(detectedObjects, now, detectionTimeout);
            }
        });
        
        
        // ============================
        // 9. Limpiar Tabla Actual
        // ============================
        document.getElementById('clearTableCurrent').addEventListener('click', () => {
            if (confirm('¿Estás seguro de que deseas limpiar las detecciones actuales?')) {
                const currentTable = document.getElementById('currentDetectionsTable');
                const totalObjectsCurrent = document.getElementById('totalObjectsCurrent');
                const rowsCleared = currentTable.rows.length;

                currentTable.innerHTML = '';
                totalObjectsCurrent.textContent = '0';

                // Reiniciar la lista de detecciones únicas
                detectedObjects.length = 0;
                console.log('Lista de objetos detectados reiniciada.');

                alert(`Detecciones actuales limpiadas. Se eliminaron ${rowsCleared} filas.`);
            }
        });

        // ============================
        // 10. Descargar Reporte
        // ============================
        document.getElementById('downloadReport').addEventListener('click', async () => {
            try {
                const table = document.getElementById('currentDetectionsTable');
                const rows = Array.from(table.querySelectorAll('tr'));
                if (rows.length === 0) {
                    alert("No hay detecciones para generar el reporte.");
                    return;
                }

                // Construir objeto con datos de la tabla
                const detections = rows.map(row => {
                    const cells = row.querySelectorAll('td');
                    return {
                        id: cells[0]?.textContent,
                        filename: cells[1]?.textContent,
                        location: cells[2]?.textContent,
                        vehicle_type: cells[3]?.textContent,
                        confidence: cells[4]?.textContent,
                        timestamp: cells[5]?.textContent,
                        processing_time: cells[6]?.textContent,
                        processed: cells[7]?.textContent,
                        bbox: cells[8]?.textContent,
                    };
                });

                // Llamar a tu backend para generar reporte
                const response = await axios.post('/generate_report', { detections });
                const link = document.createElement('a');
                link.href = response.data.report_path;  // Asegúrate de que tu backend responda con la ruta o base64
                link.download = 'reporte_detecciones.txt';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);

                alert("Reporte generado y descargado.");
            } catch (error) {
                console.error("Error al generar el reporte:", error);
                alert("Error al generar el reporte.");
            }
        });

        // ============================
        // 11. Listener para actualización de detecciones (opcional)
        // ============================
        socket.on('detection_update', (detection) => {
            addToCurrentTable({
                id: detection.id,
                filename: detection.filename,
                location: detection.location,
                vehicle_type: detection.vehicle_type,
                confidence: detection.confidence,
                bbox: detection.bbox,
                timestamp: detection.timestamp,
                processing_time: detection.processing_time,
                processed: detection.processed,
            });
        });