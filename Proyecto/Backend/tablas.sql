CREATE TABLE detections (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(256),
    location VARCHAR(512),
    vehicle_type VARCHAR(64),
    confidence FLOAT,
    timestamp TIMESTAMP,
    processing_time FLOAT,
    processed BOOLEAN,
    bbox TEXT
);

