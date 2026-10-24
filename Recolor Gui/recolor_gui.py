#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Remove COMPLETAMENTE a tonalidade azul de sprites PNG e TGA.

Uso no Windows:
    py -m pip install numpy pillow
    py recolor_gui_v2.py "C:\\Jogo\\Interface" -o "C:\\Jogo\\Interface_Cinza"
    py recolor_gui_v2.py "C:\\Jogo\\Interface" -o "C:\\Jogo\\Interface_Cinza" --mode cool

Principio: nao fazemos misturas parciais de RGB azul com cinza.
O pixel identificado como azul recebe R=G=B, de modo que nao sobra
nenhuma saturacao azul. Outros pixels sao copiados sem alteracao.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

EXTENSOES = {".png", ".tga"}


def recolorir(imagem: Image.Image, mode: str = "blue", cyan_margin: int = 3):
    """Retorna (nova_imagem, quantidade_alterada, quantidade_azul_restante).

    - blue: B > R e G <= B + margem. Inclui azul escuro e azul quase cinza.
      A tolerancia de 3 detecta pixels anti-aliased ligeiramente azul-ciano.
    - cool: B > R. Abrange todo o espectro frio, inclusive alguns verde-azulados.
    - all: remove toda saturacao (para sprites inteiramente monocromaticos).
    """
    tem_alfa = "A" in imagem.getbands() or "transparency" in imagem.info
    formato_pixels = "RGBA" if tem_alfa else "RGB"
    pixels = np.asarray(imagem.convert(formato_pixels), dtype=np.uint8).copy()
    orig_rgb = pixels[:, :, :3].copy()

    # int16 evita estouro de uint8 ao somar a margem.
    rgb = orig_rgb.astype(np.int16)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    if mode == "blue":
        mascara = (b > r) & (g <= b + cyan_margin)
    elif mode == "cool":
        mascara = b > r
    elif mode == "all":
        mascara = np.ones(r.shape, dtype=bool)
    else:
        raise ValueError(f"Modo nao suportado: {mode}")

    if tem_alfa:
        mascara &= pixels[:, :, 3] > 0

    # Cinza perceptivo (ITU-R BT.709), mantém volumes, texturas e sombras.
    # TODOS os canais recebem exatamente o mesmo inteiro, sem fade:
    # o vermelho, o verde e o azul de cada pixel selecionado ficam iguais.
    cinza = np.rint(
        0.2126 * rgb[:, :, 0] +
        0.7152 * rgb[:, :, 1] +
        0.0722 * rgb[:, :, 2]
    ).astype(np.uint8)

    pixels[:, :, 0][mascara] = cinza[mascara]
    pixels[:, :, 1][mascara] = cinza[mascara]
    pixels[:, :, 2][mascara] = cinza[mascara]

    alterados = int(np.count_nonzero(np.any(pixels[:, :, :3] != orig_rgb, axis=2)))
    rr, gg, bb = (pixels[:, :, c].astype(np.int16) for c in range(3))
    residuo = (bb > rr) & (gg <= bb + cyan_margin)
    if tem_alfa:
        residuo &= pixels[:, :, 3] > 0
    azul_restante = int(np.count_nonzero(residuo))
    return Image.fromarray(pixels, formato_pixels), alterados, azul_restante


def obter_arquivos(origem: Path, destino: Path):
    if origem.is_file():
        yield origem, Path(origem.name)
        return
    for caminho in sorted(origem.rglob("*")):
        if (
            caminho.is_file()
            and caminho.suffix.lower() in EXTENSOES
            and not caminho.is_relative_to(destino)
        ):
            yield caminho, caminho.relative_to(origem)


def executar():
    parser = argparse.ArgumentParser(
        description="Elimina completamente a cor azul de imagens PNG e TGA, preservando alfa."
    )
    parser.add_argument("input", type=Path, help="Imagem ou pasta com sprites PNG/TGA")
    parser.add_argument("-o", "--output", type=Path, help="Diretorio de saida (sem mexer nos originais)")
    parser.add_argument(
        "--mode", choices=["blue", "cool", "all"], default="blue",
        help="blue: apenas azul, incluindo tons quase cinza; cool: todos os frios; all: imagem inteira em cinza",
    )
    parser.add_argument(
        "--cyan-margin", type=int, default=3,
        help="Margem para tons cianos na deteccao do azul (padrao: 3)",
    )
    # Aceita comandos da versao anterior, mas NAO usa suavizacao ou fade.
    parser.add_argument("--blue-red", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--blue-green", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--softness", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--intensity", type=float, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cyan_margin < 0 or args.cyan_margin > 255:
        parser.error("--cyan-margin deve estar entre 0 e 255")

    origem = args.input.expanduser().resolve()
    if not origem.exists():
        parser.error(f"Caminho nao encontrado: {origem}")
    if origem.is_file() and origem.suffix.lower() not in EXTENSOES:
        parser.error("O arquivo deve ser .png ou .tga")

    pasta_padrao = origem.stem + "_cinza_v2" if origem.is_file() else origem.name + "_cinza_v2"
    destino = (args.output or origem.parent / pasta_padrao).expanduser().resolve()
    if destino == origem or (origem.is_file() and destino == origem.parent):
        parser.error("Escolha uma pasta de saida diferente dos arquivos originais")
    destino.mkdir(parents=True, exist_ok=True)

    if any(getattr(args, attr) is not None for attr in ["blue_red", "blue_green", "softness", "intensity"]):
        print("[AVISO] Flags antigas ignoradas: o v2 nao faz misturas parciais que deixam tons azuis.")

    total_imagens = 0
    total_pixels = 0
    total_erros = 0
    total_residuos = 0

    for caminho, relativo in obter_arquivos(origem, destino):
        saida = destino / relativo
        try:
            with Image.open(caminho) as imagem:
                resultado, alterados, residuos = recolorir(
                    imagem, mode=args.mode, cyan_margin=args.cyan_margin
                )
                saida.parent.mkdir(parents=True, exist_ok=True)

                salvar_opcoes = {}
                if caminho.suffix.lower() == ".tga":
                    # Mantem a orientacao e (se aplicavel) compressao TGA original.
                    salvar_opcoes["orientation"] = imagem.info.get("orientation", -1)
                    if imagem.info.get("compression") == "tga_rle":
                        salvar_opcoes["rle"] = True
                    formato = "TGA"
                else:
                    formato = "PNG"
                resultado.save(saida, format=formato, **salvar_opcoes)

                total_imagens += 1
                total_pixels += alterados
                total_residuos += residuos
                print(f"[OK] {relativo} | {alterados:,} pixels recoloridos | {residuos:,} azul restante")
        except (OSError, ValueError, UnidentifiedImageError) as e:
            total_erros += 1
            print(f"[ERRO] {caminho}: {e}")

    print(f"\nFinalizado: {total_imagens} arquivo(s), {total_pixels:,} pixels alterados, "
          f"{total_residuos:,} residuos azuis, {total_erros} erro(s).")
    print(f"Resultado em: {destino}")


if __name__ == "__main__":
    executar()
