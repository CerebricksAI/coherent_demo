# WI Studio — Frontend

Enterprise React UI for the Work Instructions API.

## Prerequisites

- Node.js 18+
- Backend running: `python -m api` (port 8000)

## Run (development)

```bash
cd UI
npm install
npm run dev
```

Open **http://localhost:5173**

Copy `.env.example` to `.env` — **use direct API URL** for uploads:

```
VITE_API_BASE_URL=http://localhost:8000
```

Large phone videos (MOV) often fail through the Vite proxy (`ECONNRESET`). Direct API URL fixes this.

## Production build

```bash
npm run build
npm run preview
```

Set `VITE_API_BASE_URL` to your API origin (e.g. `https://api.example.com`) before building.

## Features

- Drag-and-drop video upload
- Live WebSocket streaming of WI steps
- Pipeline progress (frames → transcribe → vision → WI → clips)
- Step list with clip playback
- Download links for work instructions & HTML report
- Job history (localStorage)
