# Tag your Images

## Tag all images in a folder:

bash
python img_tagger.py tag-all "C:\path\to\your\pictures"

(The database image_metadata_log.db will be created inside C:\path\to\your\pictures)

## Tag a single image (creates/uses DB in current dir unless specified):

bash
python img_tagger.py tag-single "C:\path\to\your\pictures\image1.jpg"

## or specify DB:

bash
python img_tagger.py tag-single "C:\path\to\image1.jpg" --db-path "C:\path\to\pictures\image_metadata_log.db"

## or force re-tagging:

bash
python img_tagger.py tag-single "C:\path\to\image1.jpg" --force --db-path "C:\path\to\pictures\image_metadata_log.db"

## Search for images by tags:

bash
python img_tagger.py search --db-path "C:\path\to\pictures\image_metadata_log.db" --tags boat harbor night

## Search for images by location keyword (in tags):

bash
python img_tagger.py search --db-path "C:\path\to\pictures\image_metadata_log.db" --location city

## View a specific record:

bash
python img_tagger.py view "C:\path\to\pictures\image1.jpg" --db-path "C:\path\to\pictures\image_metadata_log.db"

## Delete a specific record (from DB only):

bash
python img_tagger.py delete "C:\path\to\pictures\image1.jpg" --db-path "C:\path\to\pictures\image_metadata_log.db"


## Update tags for a record (in DB only):

bash
python img_tagger.py update-tags "C:\path\to\pictures\image1.jpg" --tags new_tag1 "another tag" final_tag --db-path "C:\path\to\pictures\image_metadata_log.db"

## options:
  -h, --help
  
  To see the specific options available for a particular command, you need to ask for help on that command.
  

# How to install 

## Install Ollama 

You have to install Ollama from https://ollama.com/download

## Get the model

After successfull Installation of Ollama:

ollama run llava

## Clone the Repository

Clone the Repository by using git clone or download the ZIP-File and unzip it.

## Install Python

If not yet installed install Python from https://www.python.org/downloads/ and install it.

## Install uv 

You may install from https://docs.astral.sh/uv/getting-started/installation/

or use 'pip install uv' in a terminal / powershell window.

## Open Terminal / Powershell

Open a  Terminal and walk to the folder where you cloned / unziped the Repository.

Type 'uv sync', then the needed packeges will be installed.

Then type 'python img_tagger.py -h'

## Have Fun