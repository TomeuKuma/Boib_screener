from database_manager import MongoDB
from scrape_manager import BoibScraper
from config import DB_URL, DB_NAME, COLLECTION_NAME
import warnings


def init_database(db, collection_name, init_url_id: str, echo=False):
    """ Inicia una base de datos vacía con un boletin de inicio a modo de seed'"""
    boib = BoibScraper(init_url_id)
    boib.get_data()
    db.save_data(boib.json_list, collection_name, echo=echo)
    # db.close()


def update_database(db, collection_name, echo=False):
    """ Vuelca en los atributos (listas) de la clase BulletinScraper los datos
    de todos los anuncios del rango de BOIBs que va desde el 'BoibScraper.url_id' hasta 'stop_url'"""

    # Bucle para iterar sobre cada BOIB:
    boib = BoibScraper()
    start = int(db.get_max_int(collection_name, 'URL_id')) + 1
    end = int(boib.current_bulletin_id) + 1
    if start == end:
        print('La colección de datos {} está actualizada!'.format(collection_name))
    else:
        print('Se va a escrapear desde la url_id: {} hasta la url_id: {}'.format(start, end - 1))
        for url_number in range(start, end):
            boib.url_id = url_number
            boib.url = 'https://www.caib.es/eboibfront/es/2020/' + str(url_number) + '/seccio-ii-autoritats-i-personal/473'
            print("Scrapping " + str(url_number))
            boib.get_data()
            db.save_data(boib.json_list, collection_name, echo=echo)
            boib.clear_lists()
            print(str(url_number) + " scrapped")
        print('La colección de datos {} está actualizada!'.format(collection_name))
        # db.close()


# código principal
if __name__ == "__main__":
    db = MongoDB(DB_URL, DB_NAME)

    if not db.get_all(COLLECTION_NAME):
        print('Se va a iniciar una nueva base de datos.')
        init_url_id = input("Introduce una seed de inicio (ej. '11690')")
        init_database(db, COLLECTION_NAME, init_url_id, echo=False)
        print('La colección {} se ha iniciado con el seed {}'.format(COLLECTION_NAME, init_url_id))
    else:
        #db.delete_all(COLLECTION_NAME)
        update_database(db, COLLECTION_NAME, echo=False)

