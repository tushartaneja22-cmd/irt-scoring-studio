"""Root entrypoint for Streamlit Community Cloud.

Streamlit Cloud is configured with main file path `app.py`, but the app lives at
`app/app.py`. This shim runs it with the correct __file__ so its relative paths
(engine/ imports, frozen link model) resolve. Run locally with either
`streamlit run app.py` or `streamlit run app/app.py`.
"""
import os
import runpy

_HERE = os.path.dirname(os.path.abspath(__file__))
runpy.run_path(os.path.join(_HERE, 'app', 'app.py'), run_name='__main__')
