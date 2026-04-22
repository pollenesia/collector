
import psycopg2
import base64
import time
import os
from datetime import datetime, timedelta
from PIL import Image
from pollenesia.utils import get_file_list

# Database connection parameters
DB_NAME = 'pollenesia'
DB_USER = 'pollenesia'
DB_PASS = 'pollenesia'
DB_HOST = 'localhost'
DB_PORT = '5433'

conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER,
                        password=DB_PASS, host=DB_HOST, port=DB_PORT)
cursor = conn.cursor()

# Create table if it doesn't exist
create_table_query = '''
CREATE TABLE IF NOT EXISTS images (
    timestamp BIGINT PRIMARY KEY,
    image_data TEXT NOT NULL
);
'''
cursor.execute(create_table_query)


def insert_data(timestamp, image_data):
    insert_query = '''
        INSERT INTO images (timestamp, image_data) VALUES (%s, %s) 
        ON CONFLICT (timestamp) DO UPDATE SET image_data = EXCLUDED.image_data;
    '''
    cursor.execute(insert_query, (timestamp, image_data))


def basename2unixtime(basename: str) -> int:
    ts = datetime.strptime(basename, '%y%m%d%H%M%S')
    return int(ts.timestamp())


pattern = '~/tmp/1/filtered/*.webp'
pattern = os.path.expanduser(pattern)

flist = get_file_list(pattern)
for fname in flist:
    with open(fname, 'rb') as f:
        image_data = f.read()
        base64_image_data = 'data:image/webp;base64,'
        base64_image_data += base64.b64encode(image_data).decode('utf-8')
        basename = os.path.basename(fname)
        basename = os.path.splitext(basename)[0]
        timestamp = basename2unixtime(basename)
        insert_data(timestamp, base64_image_data)

# Close connection
conn.commit()
cursor.close()
conn.close()
