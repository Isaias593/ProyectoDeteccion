from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Enum
from enum import Enum as PyEnum

# Inicialización de SQLAlchemy
db = SQLAlchemy()

# Definición de la tabla "detections"
class Detection(db.Model):
    __tablename__ = 'detections'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(256))
    location = db.Column(db.String(512))
    vehicle_type = db.Column(db.String(64))
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.DateTime)
    processing_time = db.Column(db.Float)
    processed = db.Column(db.Boolean)
    bbox = db.Column(db.String)  # Guarda las coordenadas como una cadena JSON

    def __repr__(self):
        return f"<Detection {self.vehicle_type} ({self.confidence})>"


#guardas imagenes

    
    
# Clase para gestionar la creación de tablas
class DatabaseManager:
    @staticmethod
    def init_app(app):
        """Inicializa SQLAlchemy con la aplicación."""
        db.init_app(app)

    @staticmethod
    def crear_tablas(app):
        """Crea las tablas en la base de datos asociadas al modelo."""
        with app.app_context():
            db.create_all()
            print("Tablas creadas exitosamente.")





