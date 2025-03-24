from src.utils.enviroment_utils import huggingface_use_domestic_endpoint, set_python_path

huggingface_use_domestic_endpoint()
set_python_path()
import src.web.app as app

app.run()