## Lo más práctico

Instala Python 3.10.

### Descargar

[Python 3.10 oficial](https://www.python.org/downloads/release/python-31011/?utm_source=chatgpt.com)

Descarga:

```text
Windows installer (64-bit)
```

---

## Luego

Crea entorno limpio:

```powershell
py -3.10 -m venv venv
```

Activa:

```powershell
venv\Scripts\activate
```

Instala:

```powershell
pip install mediapipe==0.10.14 opencv-python
pip install pyttsx3
```

Y ejecuta:

```powershell
python Manos.py
```

---

## Antes de reinstalar todo

Haz una prueba rápida:

```powershell
python -c "import mediapipe as mp; print(mp.__file__)"
```

Si te apunta a algo raro como:

```text
Vision_computadora\mediapipe.py
```

entonces NO es Python 3.11, sino conflicto de nombres.
