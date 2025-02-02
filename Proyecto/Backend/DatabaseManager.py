from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean
from sqlalchemy.dialects.postgresql import JSON  # Para usar JSON en PostgreSQL

# Inicialización de SQLAlchemy con mejores opciones de sesión
db = SQLAlchemy(session_options={"autocommit": False, "autoflush": False})

# Definición de la tabla "detections"
class Detection(db.Model):
    __tablename__ = 'detections'
    
    id = db.Column(Integer, primary_key=True)
    filename = db.Column(String(256))
    location = db.Column(String(512))
    vehicle_type = db.Column(String(64))
    confidence = db.Column(Float)
    timestamp = db.Column(DateTime, default=datetime.utcnow)  # Fecha automática
    processing_time = db.Column(Float)
    processed = db.Column(Boolean, default=False)
    bbox = db.Column(JSON)  # Almacena bbox como JSON nativo en PostgreSQL

    def __repr__(self):
        return f"<Detection {self.vehicle_type} ({self.confidence:.2f})>"

# Clase para gestionar la base de datos
class DatabaseManager:
    @staticmethod
    def init_app(app):
        """Inicializa SQLAlchemy con la aplicación Flask."""
        db.init_app(app)

    @staticmethod
    def crear_tablas(app):
        """Crea las tablas en la base de datos si no existen."""
        with app.app_context():
            try:
                db.create_all()
                print("✅ Tablas creadas exitosamente en PostgreSQL.")
            except Exception as e:
                print(f"❌ Error al crear las tablas: {e}")

    @staticmethod
    def test_connection():
        """Prueba la conexión a la base de datos."""
        try:
            with db.engine.connect() as connection:
                result = connection.execute("SELECT 1")
                return result.fetchone() is not None
        except Exception as e:
            print(f"❌ Error de conexión a la base de datos: {e}")
            return False







