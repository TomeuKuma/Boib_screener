from pymongo import MongoClient, DESCENDING
from typing import Dict


class MongoDB:

    def __init__(self, db_url, db_name):
        self.client = MongoClient(db_url)
        self.db = self.client[db_name]

    def save_data(self, json_list, collection_name, echo=True):
        """Guarda en mongodb atlas en formato .json el dataframe guardado en el
        contenido de la lista 'json_list'"""
        for element in json_list:
            collection = self.db[collection_name]
            object_id = collection.insert_one(element).inserted_id
            if echo:
                print(str(object_id) + ' inserted in ' + element["HTML"])

    def get_max_int(self, collection_name, field_name):
        """Consulta en mongodb atlas y devuelve en una lista de jsons con el contenido de
        la colección que se le pase como parámetro'"""
        collection = self.db[collection_name]
        last_id = collection.find().sort(field_name, DESCENDING).limit(1)[0][field_name]
        return last_id

    def get_all(self, collection_name, echo=False):
        """Consulta en mongodb atlas y devuelve en una lista de jsons con el contenido de
        la colección que se le pase como parámetro'"""
        collection = self.db[collection_name]
        cursor = collection.find()
        data_list = []
        # Imprimir cada documento en la colección
        for document in cursor:
            data_list.append(document)
            if echo:
                print(document)
        return data_list

    def delete_data(self, collection_name, data: Dict, only_one=True):
        """Elimina el registro cuyo valor se le pase como diccionario. Puede eliminar una o varias
        coincidencias con el parámetro only_one"""
        if only_one:
            collection = self.db[collection_name]
            collection.delete_one(data)
        else:
            collection = self.db[collection_name]
            collection.delete_many(data)

    def delete_all(self, collection_name):
        """Elimina todos los valores de la base de datos"""
        collection = self.db[collection_name]
        collection.delete_many({})
        print('Base de datos borrada.')
