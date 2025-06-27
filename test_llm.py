import numpy as np
import os
import dotenv

dotenv.load_dotenv()

if (np.__version__).startswith("1."):
    print("Numpy version is 1.*.*, you're good to go!")
else:
    raise ValueError("Please restart your runtime using the above instructions!")