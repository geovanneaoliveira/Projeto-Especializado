"""
Batch YOLO Inference Script
============================
Runs YOLO inference on all image files in a folder, processing them in
batches for better GPU/CPU throughput, and saves annotated results plus
a CSV summary of all detections.

Usage:
    python batch_inference.py --model best.pt --source ./images --output ./results

    # Optional flags
    python batch_inference.py --model best.pt --source ./images --output ./results \
        --batch 16 --conf 0.25 --iou 0.45 --imgsz 640 --device 0

Requirements:
    pip install ultralytics
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from time import time

try:
    from ultralytics import YOLO
except ImportError:
    print("ERRO: ultralytics não encontrado. Instale com: pip install ultralytics")
    sys.exit(1)

try:
    import cv2
    import numpy as np
except ImportError:
    print("ERRO: opencv-python/numpy não encontrados. Instale com: pip install opencv-python numpy")
    sys.exit(1)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_images(source_dir: Path) -> list[Path]:
    """Recursively find all image files in the source directory."""
    images = [
        p for p in sorted(source_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return images


def chunk_list(items: list, size: int):
    """Yield successive chunks of `size` from `items`."""
    for i in range(0, len(items), size):
        yield items[i:i + size]


def build_mosaic(
    images: list,
    tile_size: int = 320,
    max_per_mosaic: int = 25,
    cols: int | None = None,
) -> list:
    """
    Arrange a list of annotated images (numpy arrays, BGR) into one or more
    grid mosaics. Each tile is resized to a square `tile_size` x `tile_size`
    (letterboxed to preserve aspect ratio). Returns a list of mosaic images
    (numpy arrays), one per chunk of `max_per_mosaic` images.
    """
    mosaics = []

    for chunk in chunk_list(images, max_per_mosaic):
        n = len(chunk)
        grid_cols = cols if cols else math.ceil(math.sqrt(n))
        grid_rows = math.ceil(n / grid_cols)

        canvas = np.full(
            (grid_rows * tile_size, grid_cols * tile_size, 3), 30, dtype=np.uint8
        )

        for idx, img in enumerate(chunk):
            row, col = divmod(idx, grid_cols)

            h, w = img.shape[:2]
            scale = min(tile_size / w, tile_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            tile = np.full((tile_size, tile_size, 3), 30, dtype=np.uint8)
            y_off = (tile_size - new_h) // 2
            x_off = (tile_size - new_w) // 2
            tile[y_off:y_off + new_h, x_off:x_off + new_w] = resized

            y0, y1 = row * tile_size, (row + 1) * tile_size
            x0, x1 = col * tile_size, (col + 1) * tile_size
            canvas[y0:y1, x0:x1] = tile

        mosaics.append(canvas)

    return mosaics


def run_batch_inference(
    model_path: str,
    source_dir: str,
    output_dir: str,
    batch_size: int = 16,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    device: str = "cpu",
    save_images: bool = True,
    show_mosaic: bool = True,
    mosaic_tile_size: int = 320,
    mosaic_max_per_grid: int = 25,
    mosaic_cols: int | None = None,
):
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        print(f"ERRO: pasta de origem não encontrada: {source_path}")
        sys.exit(1)

    print(f"Carregando modelo: {model_path}")
    model = YOLO(model_path)

    images = find_images(source_path)
    if not images:
        print(f"Nenhuma imagem encontrada em: {source_path}")
        sys.exit(0)

    print(f"Encontradas {len(images)} imagens. Processando em lotes de {batch_size}...")

    csv_path = output_path / "detections.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["image", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]
    )

    total_detections = 0
    total_images_processed = 0
    annotated_frames = []  # collected for mosaic building
    start_time = time()

    for batch_idx, batch_paths in enumerate(chunk_list(images, batch_size), start=1):
        batch_str_paths = [str(p) for p in batch_paths]

        results = model.predict(
            source=batch_str_paths,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            save=save_images,
            project=str(output_path),
            name="predictions",
            exist_ok=True,
            verbose=False,
        )

        for img_path, result in zip(batch_paths, results):
            n_detections = len(result.boxes) if result.boxes is not None else 0
            total_detections += n_detections

            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    cls_name = result.names.get(cls_id, str(cls_id))
                    confidence = float(box.conf.item())
                    x1, y1, x2, y2 = [round(v, 2) for v in box.xyxy[0].tolist()]
                    csv_writer.writerow(
                        [img_path.name, cls_id, cls_name, round(confidence, 4), x1, y1, x2, y2]
                    )

            if show_mosaic:
                # result.plot() returns an annotated BGR numpy array (boxes/labels drawn)
                annotated_frames.append(result.plot())

            total_images_processed += 1

        print(
            f"  Lote {batch_idx}: {len(batch_paths)} imagens processadas "
            f"({total_images_processed}/{len(images)} total)"
        )

    csv_file.close()
    elapsed = time() - start_time

    mosaic_paths = []
    if show_mosaic and annotated_frames:
        print(f"\nMontando mosaico(s) com {len(annotated_frames)} imagens...")
        mosaics = build_mosaic(
            annotated_frames,
            tile_size=mosaic_tile_size,
            max_per_mosaic=mosaic_max_per_grid,
            cols=mosaic_cols,
        )
        for idx, mosaic in enumerate(mosaics, start=1):
            mosaic_path = output_path / f"mosaic_{idx:02d}.jpg"
            cv2.imwrite(str(mosaic_path), mosaic)
            mosaic_paths.append(mosaic_path)
            print(f"  Mosaico salvo: {mosaic_path}")

        # Try to display interactively if a display is available
        try:
            for idx, mosaic in enumerate(mosaics, start=1):
                window_name = f"Resultados - Mosaico {idx}/{len(mosaics)}"
                cv2.imshow(window_name, mosaic)
            print("\nPressione qualquer tecla em uma janela de imagem para fechar...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            print("(Sem display disponível para exibição interativa — mosaicos salvos em disco apenas)")

    print("\n" + "=" * 50)
    print("Inferência concluída")
    print("=" * 50)
    print(f"Imagens processadas : {total_images_processed}")
    print(f"Detecções totais    : {total_detections}")
    print(f"Tempo total         : {elapsed:.2f}s")
    print(f"Tempo médio/imagem  : {elapsed / max(total_images_processed, 1):.3f}s")
    print(f"CSV de detecções    : {csv_path}")
    if save_images:
        print(f"Imagens anotadas    : {output_path / 'predictions'}")
    if mosaic_paths:
        print(f"Mosaico(s)          : {', '.join(str(p) for p in mosaic_paths)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Roda inferência YOLO em lote em todas as imagens de uma pasta."
    )
    parser.add_argument("--model", required=True, help="Caminho para o arquivo .pt do modelo")
    parser.add_argument("--source", required=True, help="Pasta com as imagens de entrada")
    parser.add_argument("--output", default="./output", help="Pasta de saída (default: ./output)")
    parser.add_argument("--batch", type=int, default=16, help="Tamanho do lote (default: 16)")
    parser.add_argument("--conf", type=float, default=0.25, help="Limiar de confiança (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="Limiar de IoU para NMS (default: 0.45)")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamanho da imagem para inferência (default: 640)")
    parser.add_argument("--device", default="cpu", help="Device: 'cpu', '0', '0,1', etc. (default: cpu)")
    parser.add_argument(
        "--no-save-images", action="store_true",
        help="Não salvar as imagens anotadas, apenas o CSV de detecções"
    )
    parser.add_argument(
        "--no-mosaic", action="store_true",
        help="Não gerar/exibir o mosaico com todos os resultados"
    )
    parser.add_argument(
        "--mosaic-tile-size", type=int, default=320,
        help="Tamanho (px) de cada tile quadrado no mosaico (default: 320)"
    )
    parser.add_argument(
        "--mosaic-max-per-grid", type=int, default=25,
        help="Máximo de imagens por mosaico antes de criar um novo (default: 25)"
    )
    parser.add_argument(
        "--mosaic-cols", type=int, default=None,
        help="Número fixo de colunas no grid (default: automático, próximo de um quadrado)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_batch_inference(
        model_path=args.model,
        source_dir=args.source,
        output_dir=args.output,
        batch_size=args.batch,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save_images=not args.no_save_images,
        show_mosaic=not args.no_mosaic,
        mosaic_tile_size=args.mosaic_tile_size,
        mosaic_max_per_grid=args.mosaic_max_per_grid,
        mosaic_cols=args.mosaic_cols,
    )
