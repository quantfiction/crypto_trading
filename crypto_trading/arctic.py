import arcticdb as adb
from pathlib import Path

class ArcticHandler:
    """Handler for Arctic data storage"""

    def __init__(self, db_name):
        self.db_name = db_name
        self.arctic_path = self.create_arctic_string(db_name)
        self.arctic_db = adb.Arctic(self.arctic_path)

    def create_arctic_string(self, db_name):
        home_dir = Path.home()
        arctic_path = home_dir / db_name
        str_arctic_path = 'lmdb:///' + str(arctic_path.resolve())
        
        return str_arctic_path

    def set_lib(self, lib_name):
        if lib_name not in self.arctic_db.list_libraries():
            self.arctic_db.create_library(lib_name)
        self.lib = self.arctic_db.get_library(lib_name)

    def get_lib(self, lib_name):
        if lib_name not in self.arctic_db.list_libraries():
            self.arctic_db.create_library(lib_name)
        return self.arctic_db.get_library(lib_name)

