#!/usr/bin/env python
"""Captura de la ventana de la aplicación para los reportes JPG/PDF.

tkcap (y pyautogui por debajo) toman una foto de todo el escritorio y la
recortan a la región donde *debería* estar la ventana. Si en el instante
exacto de la captura la ventana está tapada por otra, sin foco, o la
pantalla se bloqueó/apagó, el recorte sale negro en vez del contenido real.

En Windows se evita ese problema con la API PrintWindow, que le pide a la
ventana que renderice su propio contenido directamente en un bitmap,
independientemente de lo que esté visible en el escritorio en ese momento.
"""

import sys

from PIL import Image


def capture_window(root):
    """Captura la ventana `root` y devuelve una imagen PIL."""
    if sys.platform == "win32":
        return _capture_window_win32(root)
    return _capture_window_fallback(root)


def _capture_window_win32(root):
    """Renderiza la ventana con PrintWindow (PW_RENDERFULLCONTENT), inmune
    a oclusión, pérdida de foco o bloqueo de pantalla."""
    import ctypes

    import win32gui
    import win32ui

    hwnd = win32gui.FindWindow(None, root.title())
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width, height = right - left, bottom - top

    hwnd_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()
    bitmap = win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
    save_dc.SelectObject(bitmap)

    PW_RENDERFULLCONTENT = 0x00000002
    ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), PW_RENDERFULLCONTENT)

    bmp_info = bitmap.GetInfo()
    bmp_bits = bitmap.GetBitmapBits(True)
    img = Image.frombuffer(
        "RGB",
        (bmp_info["bmWidth"], bmp_info["bmHeight"]),
        bmp_bits,
        "raw",
        "BGRX",
        0,
        1,
    )

    win32gui.DeleteObject(bitmap.GetHandle())
    save_dc.DeleteDC()
    mfc_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwnd_dc)
    return img


def _capture_window_fallback(root):
    """Captura por recorte de pantalla (comportamiento previo), usada en
    plataformas donde PrintWindow no aplica."""
    import pyautogui
    import tkcap

    region = tkcap.CAP(root).get_region()
    return pyautogui.screenshot(region=tuple(region))
