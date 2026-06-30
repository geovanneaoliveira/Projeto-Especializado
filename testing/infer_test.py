"""
infer_test.py — Inferência simples com YOLO em arquivo local (sem GUI)
Uso: python infer_test.py --model best.pt --source imagem.jpg [--conf 0.25] [--iou 0.45] [--show]
"""

# python testing/infer_test.py --model .\models\best\cocacola_yolo26m_inspecao_v1_best.pt --source .\images\IMG-20250630-WA0173.jpg --show

import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERRO] ultralytics não instalado. Execute: pip install ultralytics")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("[ERRO] opencv-python não instalado. Execute: pip install opencv-python")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Inferência YOLO em arquivo local")
    parser.add_argument("--model",  required=True, help="Caminho para o modelo (.pt)")
    parser.add_argument("--source", required=True, help="Imagem ou vídeo de entrada")
    parser.add_argument("--conf",   type=float, default=0.25, help="Limiar de confiança (padrão: 0.25)")
    parser.add_argument("--iou",    type=float, default=0.45, help="Limiar de IoU para NMS (padrão: 0.45)")
    parser.add_argument("--imgsz",  type=int,   default=640,  help="Tamanho da imagem (padrão: 640)")
    parser.add_argument("--show",   action="store_true",      help="Exibir resultado na tela com cv2.imshow")
    parser.add_argument("--save",   action="store_true",      help="Salvar resultado em runs/detect/")
    return parser.parse_args()


def run_image(model, source_path: Path, args):
    print(f"\n[INFO] Rodando inferência em: {source_path}")

    results = model.predict(
        source=str(source_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        save=args.save,
        verbose=False,
    )

    result = results[0]
    boxes  = result.boxes

    print(f"\n{'='*50}")
    print(f"  Arquivo : {source_path.name}")
    print(f"  Detecções encontradas: {len(boxes)}")
    print(f"{'='*50}")

    if len(boxes) == 0:
        print("  Nenhum objeto detectado.\n")
    else:
        print(f"  {'#':<4} {'Classe':<20} {'Confiança':>10}  {'BBox (x1,y1,x2,y2)'}")
        print(f"  {'-'*60}")
        for i, box in enumerate(boxes):
            cls_id  = int(box.cls[0])
            cls_name = result.names[cls_id]
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            print(f"  {i:<4} {cls_name:<20} {conf_val:>10.2%}  ({x1}, {y1}, {x2}, {y2})")
        print()

    if args.show:
        annotated = result.plot()
        cv2.imshow("YOLO — Resultado", annotated)
        print("[INFO] Pressione qualquer tecla na janela para fechar.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result


def run_video(model, source_path: Path, args):
    print(f"\n[INFO] Rodando inferência em vídeo: {source_path}")

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir o vídeo: {source_path}")
        sys.exit(1)

    total_frames     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_detections = 0
    frame_idx        = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            save=False,
            verbose=False,
        )

        result = results[0]
        n_det  = len(result.boxes)
        total_detections += n_det

        print(f"  Frame {frame_idx+1:>5}/{total_frames}  — {n_det} detecção(ões)", end="\r")

        if args.show:
            annotated = result.plot()
            cv2.imshow("YOLO — Vídeo", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[INFO] Interrompido pelo usuário.")
                break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n\n{'='*50}")
    print(f"  Arquivo : {source_path.name}")
    print(f"  Frames processados : {frame_idx}")
    print(f"  Total de detecções : {total_detections}")
    print(f"  Média por frame    : {total_detections/max(frame_idx,1):.2f}")
    print(f"{'='*50}\n")


def main():
    args = parse_args()

    model_path  = Path(args.model)
    source_path = Path(args.source)

    if not model_path.exists():
        print(f"[ERRO] Modelo não encontrado: {model_path}")
        sys.exit(1)

    if not source_path.exists():
        print(f"[ERRO] Arquivo de entrada não encontrado: {source_path}")
        sys.exit(1)

    print(f"[INFO] Carregando modelo: {model_path}")
    model = YOLO(str(model_path))
    print(f"[INFO] Classes do modelo: {list(model.names.values())}")
    print(f"[INFO] conf={args.conf}  iou={args.iou}  imgsz={args.imgsz}")

    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    ext = source_path.suffix.lower()

    if ext in IMAGE_EXTS:
        run_image(model, source_path, args)
    elif ext in VIDEO_EXTS:
        run_video(model, source_path, args)
    else:
        print(f"[ERRO] Extensão '{ext}' não suportada. Use imagem ou vídeo.")
        sys.exit(1)


if __name__ == "__main__":
    main()
