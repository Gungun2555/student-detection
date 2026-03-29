# Student Face Recognition System

Production-grade pipeline: RetinaFace + ArcFace + FAISS + GFPGAN

## Architecture

```
project/
├── services/
│   ├── detector.py      # RetinaFace face detection
│   ├── aligner.py       # 5-point landmark alignment
│   ├── enhancer.py      # GFPGAN enhancement (low-quality faces)
│   ├── embedder.py      # ArcFace 512-d embeddings
│   ├── enrollment.py    # Enrollment pipeline + augmentation
│   └── recognition.py  # Recognition pipeline
├── database/
│   └── faiss_db.py      # FAISS vector DB + voting
├── api/
│   └── routes.py        # FastAPI endpoints
├── utils/
│   ├── image_utils.py
│   └── logger.py
├── models/              # Place GFPGANv1.4.pth here
├── enroll_students.py   # Bulk enrollment script
├── recognize_photos.py  # Batch recognition script
└── main.py              # API server
```

## Setup

### 1. Create & activate virtual environment

```bash
cd project

# Windows (one-time setup — creates venv + installs all deps)
setup.bat

# Manual activation for subsequent sessions
venv\Scripts\activate.bat

# Linux / macOS
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> CPU only? Before running setup, edit `requirements.txt`:
>
> - `onnxruntime-gpu` → `onnxruntime`
> - `faiss-gpu` → `faiss-cpu`

### 2. Download GFPGAN model (optional but recommended)

```bash
mkdir -p project/models
curl -L https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth \
     -o project/models/GFPGANv1.4.pth
```

InsightFace (buffalo_l) downloads automatically on first run.

### 3. Enroll students

```bash
cd project

# Windows shortcut (auto-activates venv)
enroll.bat
enroll.bat --no-augment

# Or with venv active manually
python enroll_students.py
```

### 4. Recognize group photos

```bash
# Windows shortcut
recognize.bat --save-json

# Or manually
python recognize_photos.py --save-json
```

Annotated images are saved to `project/output/`.

### 5. Start the API server

```bash
# Windows shortcut
start_api.bat

# Or manually
python main.py
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

## API Endpoints

| Method | Endpoint                  | Description                        |
| ------ | ------------------------- | ---------------------------------- |
| POST   | `/api/v1/enroll`          | Enroll a student                   |
| POST   | `/api/v1/recognize`       | Recognize faces in one image       |
| POST   | `/api/v1/batch-recognize` | Recognize faces in multiple images |
| GET    | `/api/v1/db/stats`        | Database statistics                |

### Example: Enroll

```bash
curl -X POST http://localhost:8000/api/v1/enroll \
  -F "student_id=Chandan B L" \
  -F "images=@data/students/1-Chandan B L - Photo.jpg"
```

### Example: Recognize

```bash
curl -X POST http://localhost:8000/api/v1/recognize \
  -F "image=@data/PHOTO 1.jpeg"
```

Response:

```json
{
  "faces": [
    {
      "bbox": [120, 45, 210, 160],
      "name": "Chandan B L",
      "confidence": 0.82,
      "quality": "medium",
      "det_score": 0.97
    },
    {
      "bbox": [300, 60, 390, 175],
      "name": "Unknown",
      "confidence": 0.31,
      "quality": "low",
      "det_score": 0.88
    }
  ],
  "total": 2
}
```

## Docker

```bash
docker build -t face-recog .
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/data:/app/../data \
  -v $(pwd)/project/database:/app/database \
  face-recog
```

## Matching Thresholds

| Quality | Threshold |
| ------- | --------- |
| High    | 0.60      |
| Medium  | 0.50      |
| Low     | 0.40      |

Quality is auto-detected per face using blur score + resolution.


cd project
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Invoke-WebRequest -Uri "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth" -OutFile "models\GFPGANv1.4.pth"
python enroll_students.py
python recognize_photos.py --save-json
