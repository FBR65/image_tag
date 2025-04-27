import os
import requests
import base64
import time
import sqlite3
from datetime import datetime
import json
import argparse  # For CLI arguments

# Metadata handling libraries
from PIL import Image, UnidentifiedImageError, ExifTags
import piexif
import piexif.helper

# --- New Imports for GPS/Location ---
import exifread  # For robust GPS tag reading
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
# --- End New Imports ---


# --- Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llava"
DEFAULT_DB_NAME = "image_metadata_log.db"
# --- End Configuration ---


# Helper to make EXIF data JSON serializable
def _sanitize_for_json(data):
    if isinstance(data, bytes):
        # Try decoding bytes, fallback to base64 representation
        try:
            return data.decode("utf-8", errors="replace")
        except UnicodeDecodeError:
            # For non-text bytes (like thumbnails or unknown data)
            return f"base64:{base64.b64encode(data).decode('ascii')}"
    elif isinstance(data, dict):
        return {k: _sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_sanitize_for_json(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(_sanitize_for_json(item) for item in data)
    # Handle other non-serializable types if necessary
    elif isinstance(data, (int, float, str, bool, type(None))):
        return data
    else:
        # Fallback for unknown types
        return repr(data)


# Helper to get human-readable EXIF tag names
def _get_exif_with_names(exif_dict):
    exif_with_names = {}
    for ifd_name, ifd_dict in exif_dict.items():
        if ifd_name == "thumbnail":
            exif_with_names[ifd_name] = (
                "thumbnail_data"  # Avoid storing large blob directly
            )
            continue
        if not isinstance(ifd_dict, dict):
            continue

        exif_with_names[ifd_name] = {}
        for tag, value in ifd_dict.items():
            tag_name = ExifTags.TAGS.get(tag, tag)  # Get name if available
            # Special handling for GPS IFD tags
            if ifd_name == "GPS":
                gps_tag_name = ExifTags.GPSTAGS.get(tag, tag)
                exif_with_names[ifd_name][gps_tag_name] = value
            else:
                exif_with_names[ifd_name][tag_name] = (
                    value  # Keep original value for now
                )

    return exif_with_names


class ImageMetadataManager:
    """
    Manages image metadata tagging, storage, and retrieval using Ollama and SQLite.
    Includes CLI interaction logic and GPS-based location tagging.
    """

    def __init__(self, root_folder: str | None = None, db_path: str | None = None):
        """
        Initializes the ImageMetadataManager.

        Args:
            root_folder: The starting directory for batch processing. Required for 'tag-all'.
            db_path: Explicit path to the database file. If None, it's derived from root_folder or CWD.

        Raises:
            ValueError: If root_folder is required but invalid/not provided.
            ConnectionError: If database initialization fails.
        """
        self.root_folder = None
        self.db_path = db_path

        if root_folder:
            if not os.path.isdir(root_folder):
                raise ValueError(
                    f"Root folder '{root_folder}' does not exist or is not a directory."
                )
            self.root_folder = os.path.abspath(root_folder)
            # Default DB location is inside the root folder if not specified
            if not self.db_path:
                self.db_path = os.path.join(self.root_folder, DEFAULT_DB_NAME)
        elif not self.db_path:
            # If no root folder and no db_path, use CWD for DB (e.g., for single file ops)
            self.db_path = os.path.join(os.getcwd(), DEFAULT_DB_NAME)

        # Ensure the directory for the database exists
        db_dir = os.path.dirname(self.db_path)
        if not os.path.exists(db_dir):
            try:
                os.makedirs(db_dir)
                print(f"Created directory for database: {db_dir}")
            except OSError as e:
                raise ConnectionError(
                    f"Failed to create directory for database '{db_dir}': {e}"
                ) from e

        self.supported_extensions = (
            ".jpg",
            ".jpeg",
            ".png",  # Note: PNG typically doesn't store standard EXIF/GPS
            ".gif",
            ".bmp",
            ".tiff",
            ".webp",
        )
        self.conn = None
        self.cursor = None
        # --- Initialize Geocoder ---
        # Use a descriptive user_agent as required by Nominatim's ToS
        self.geolocator = Nominatim(user_agent="image_tagger_script_v1.0")
        # --- End Geocoder Init ---
        self._initialize_database()
        if self.root_folder:
            print(f"Initialized Manager for root folder: {self.root_folder}")
        print(f"Using database: {self.db_path}")

    def _initialize_database(self):
        """Connects to or creates the SQLite database and table."""
        try:
            # Use WAL mode for potentially better concurrency, though likely not critical here
            self.conn = sqlite3.connect(
                self.db_path, isolation_level=None
            )  # Autocommit mode
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.cursor = self.conn.cursor()
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS processed_images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_path TEXT NOT NULL UNIQUE,
                    ollama_tags TEXT,       -- JSON list of tags from Ollama (may include location)
                    all_exif_data TEXT,     -- JSON blob of all extracted EXIF/metadata
                    processed_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    status TEXT,            -- e.g., 'Success', 'No Tags', 'Metadata Error', 'Ollama Error', 'GPS Error'
                    error_message TEXT NULL
                )
            """)
            # Consider adding indexes for faster searching if needed
            # self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_ollama_tags ON processed_images(ollama_tags);")
            print("Database initialized successfully.")
        except sqlite3.Error as e:
            print(f"Error initializing database '{self.db_path}': {e}")
            raise ConnectionError(f"Failed to initialize database: {e}") from e

    def _is_already_processed(self, image_path: str) -> bool:
        """Checks if the image path already exists in the database."""
        # This check is now primarily for the batch 'tag-all' command.
        # 'tag-single' will overwrite.
        if not self.cursor:
            return False
        try:
            self.cursor.execute(
                "SELECT 1 FROM processed_images WHERE original_path = ?", (image_path,)
            )
            return self.cursor.fetchone() is not None
        except sqlite3.Error as e:
            print(f"Error checking database for {os.path.basename(image_path)}: {e}")
            return False

    def _log_to_database(
        self,
        original_path: str,
        ollama_tags: list[str] | None,
        all_metadata: dict | None,  # Renamed from all_exif for clarity
        status: str,
        error_msg: str | None = None,
    ):
        """Logs or updates the processing result in the database."""
        if not self.conn or not self.cursor:
            print("Error: Database connection not available for logging.")
            return

        tags_json = json.dumps(ollama_tags) if ollama_tags else None

        # Sanitize metadata before converting to JSON
        metadata_json = (
            json.dumps(_sanitize_for_json(all_metadata)) if all_metadata else None
        )
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            self.cursor.execute(
                """
                INSERT INTO processed_images (original_path, ollama_tags, all_exif_data, processed_timestamp, status, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(original_path) DO UPDATE SET
                    ollama_tags=excluded.ollama_tags,
                    all_exif_data=excluded.all_exif_data,
                    processed_timestamp=excluded.processed_timestamp,
                    status=excluded.status,
                    error_message=excluded.error_message
            """,
                (original_path, tags_json, metadata_json, timestamp, status, error_msg),
            )
            # No explicit commit needed due to isolation_level=None
        except sqlite3.Error as e:
            print(
                f"Error logging to database for {os.path.basename(original_path)}: {e}"
            )
            # Rollback isn't applicable in autocommit mode

    def _get_tags_from_ollama(
        self, image_path: str
    ) -> tuple[list[str] | None, str | None]:
        """Gets tags for the image using the Ollama API."""
        print(
            f"  Attempting to get tags from Ollama for: {os.path.basename(image_path)}..."
        )
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            prompt = "Generate 5-7 concise keywords describing this image, separated by commas. Focus on objects, scene, and concepts."
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "images": [encoded_string],
                "stream": False,
            }
            # Increased timeout for potentially slower model processing
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=180)
            response.raise_for_status()
            response_data = response.json()
            raw_tags_text = response_data.get("response", "").strip()

            if not raw_tags_text:
                return None, "Ollama returned empty response"

            tags = [
                tag.strip().lower() for tag in raw_tags_text.split(",") if tag.strip()
            ]
            tags = list(dict.fromkeys(tags))  # Remove duplicates

            print(f"  Ollama tags received: {tags}")
            return (
                (tags, None)
                if tags
                else (None, "Ollama response parsed to empty tag list")
            )

        except requests.exceptions.Timeout:
            return None, "Ollama API request timed out"
        except requests.exceptions.RequestException as e:
            return None, f"Ollama API connection error: {e}"
        except FileNotFoundError:
            return None, f"Image file not found at {image_path}"
        except KeyError:
            return None, "Unexpected Ollama response format"
        except Exception as e:
            return None, f"Ollama interaction error: {e}"

    # --- New GPS Helper Methods ---
    def _convert_to_degrees(self, value):
        """Helper function to convert the GPS coordinates stored in the EXIF to degrees"""
        # Check if the value is valid and has the expected structure
        if not hasattr(value, "__len__") or len(value) < 3:
            return None  # Invalid format

        try:
            d = (
                float(value[0].num) / float(value[0].den)
                if value[0].den != 0
                else float(value[0].num)
            )
            m = (
                float(value[1].num) / float(value[1].den)
                if value[1].den != 0
                else float(value[1].num)
            )
            s = (
                float(value[2].num) / float(value[2].den)
                if value[2].den != 0
                else float(value[2].num)
            )
            return d + (m / 60.0) + (s / 3600.0)
        except (ZeroDivisionError, AttributeError, IndexError, ValueError) as e:
            print(f"    Error converting GPS rational to degrees: {e}")
            return None  # Error during conversion

    def _get_gps_coordinates(self, image_path: str) -> tuple[float, float] | None:
        """Extracts GPS coordinates using exifread."""
        print(f"  Checking for GPS data in: {os.path.basename(image_path)}")
        try:
            with open(image_path, "rb") as f:
                tags = exifread.process_file(
                    f, stop_tag="GPS GPSLongitude"
                )  # Stop early for efficiency

            if not tags:
                print("    No EXIF tags found by exifread.")
                return None

            gps_latitude = tags.get("GPS GPSLatitude")
            gps_latitude_ref = tags.get("GPS GPSLatitudeRef")
            gps_longitude = tags.get("GPS GPSLongitude")
            gps_longitude_ref = tags.get("GPS GPSLongitudeRef")

            if (
                gps_latitude
                and gps_latitude_ref
                and gps_longitude
                and gps_longitude_ref
            ):
                lat = self._convert_to_degrees(gps_latitude.values)
                lon = self._convert_to_degrees(gps_longitude.values)

                if lat is None or lon is None:
                    print("    Failed to convert GPS coordinates.")
                    return None  # Conversion error

                # Check orientation (North/South, East/West)
                if gps_latitude_ref.values[0] == "S":
                    lat = -lat
                if gps_longitude_ref.values[0] == "W":
                    lon = -lon

                print(f"    Found GPS Coordinates: Lat {lat:.5f}, Lon {lon:.5f}")
                return lat, lon
            else:
                print("    Required GPS tags not found.")
                return None
        except FileNotFoundError:
            print(f"    Error: File not found during GPS read: {image_path}")
            return None
        except Exception as e:
            print(f"    Error reading GPS data with exifread: {e}")
            return None

    def _get_location_from_coordinates(
        self, latitude: float, longitude: float
    ) -> str | None:
        """Gets a location name (full address) from GPS coordinates using geopy."""
        print(
            f"  Attempting reverse geocoding for Lat {latitude:.5f}, Lon {longitude:.5f}..."
        )
        try:
            # timeout parameter is important for network issues
            # language='en' helps ensure consistent results
            location = self.geolocator.reverse(
                (latitude, longitude), exactly_one=True, language="en", timeout=10
            )
            if location and location.address:
                # --- Correction: Return the full address ---
                full_address = location.address.strip()
                print(f"    Reverse geocoding successful: {full_address}")
                # Return the full address string as the tag
                # Convert to lowercase for consistency with other tags
                return full_address.lower()
                # --- End Correction ---
            else:
                print("    Reverse geocoding returned no result.")
                return None
        except GeocoderTimedOut:
            print("    Error: Reverse geocoding service timed out.")
            return None
        except GeocoderServiceError as e:
            print(f"    Error: Reverse geocoding service error: {e}")
            return None
        except Exception as e:
            print(f"    Unexpected error during reverse geocoding: {e}")
            return None

    # --- End GPS Helper Methods ---

    def _extract_and_add_metadata(
        self, image_path: str, tags: list[str]
    ) -> tuple[dict | None, bool, str | None]:
        """
        Extracts existing metadata, adds combined (Ollama + Location) tags
        (to UserComment for EXIF, or a text chunk for PNG), and returns all extracted metadata.

        Returns:
            A tuple: (all_extracted_metadata_dict or None, success_writing_tags boolean, error_message string or None)
        """
        print(f"  Processing metadata for: {os.path.basename(image_path)}")
        # Join the final list of tags (Ollama + potentially location)
        tags_str = ", ".join(sorted(list(set(tags))))  # Sort and ensure unique
        print(f"  Final tags to write: {tags_str}")

        extracted_metadata = None
        write_success = False
        error_msg = None

        try:
            # Use 'with' statement for reliable file handling with Pillow
            with Image.open(image_path) as img:
                img_format = img.format
                # --- JPEG/TIFF Handling (Existing Logic) ---
                if img_format in ["JPEG", "TIFF"]:
                    try:
                        # Load existing EXIF using piexif
                        exif_dict = piexif.load(img.info.get("exif", b""))
                        # Get human-readable names for storage/viewing
                        extracted_metadata = _get_exif_with_names(exif_dict)
                    except Exception as e:  # Catch broader errors during load
                        print(
                            f"  Warning: Problem loading existing EXIF for {os.path.basename(image_path)} ({e}). Creating new structure."
                        )
                        exif_dict = {
                            "0th": {},
                            "Exif": {},
                            "GPS": {},
                            "1st": {},
                            "thumbnail": None,
                        }
                        # Try to get basic info if EXIF fails
                        extracted_metadata = (
                            {"Info": img.info.copy()} if img.info else {}
                        )

                    try:
                        # Ensure Exif IFD exists
                        if "Exif" not in exif_dict:
                            exif_dict["Exif"] = {}
                        # Add tags to UserComment
                        exif_dict["Exif"][piexif.ExifIFD.UserComment] = (
                            piexif.helper.UserComment.dump(tags_str, encoding="unicode")
                        )
                        exif_bytes = piexif.dump(exif_dict)
                        # Save back using Pillow's save method with exif bytes
                        # Preserve quality settings if possible
                        save_kwargs = {"exif": exif_bytes}
                        if img_format == "JPEG":
                            save_kwargs["quality"] = img.info.get(
                                "quality", 95
                            )  # Default to 95 if not found
                            save_kwargs["subsampling"] = img.info.get(
                                "subsampling", -1
                            )  # Preserve subsampling
                            save_kwargs["progressive"] = img.info.get(
                                "progressive", False
                            )
                            save_kwargs["icc_profile"] = img.info.get(
                                "icc_profile"
                            )  # Preserve color profile

                        img.save(image_path, **save_kwargs)
                        print("  Successfully added tags to EXIF UserComment.")
                        write_success = True
                        # Update extracted_metadata with the UserComment we just added for logging consistency
                        if "Exif" not in extracted_metadata:
                            extracted_metadata["Exif"] = {}
                        extracted_metadata["Exif"]["UserComment"] = (
                            tags_str  # Store the readable string
                        )

                    except Exception as write_e:
                        error_msg = f"Failed to write EXIF tags: {write_e}"
                        print(f"  Error: {error_msg}")

                # --- PNG Handling ---
                elif img_format == "PNG":
                    print(
                        f"  Processing PNG metadata for: {os.path.basename(image_path)}"
                    )
                    try:
                        # 1. Read existing metadata (text chunks are in img.info)
                        existing_info = img.info or {}
                        extracted_metadata = {
                            "Info": existing_info.copy()
                        }  # Store existing info
                        print(
                            f"  Extracted existing PNG info keys: {list(existing_info.keys())}"
                        )

                        # 2. Prepare new metadata using PngInfo object
                        from PIL.PngImagePlugin import PngInfo

                        pnginfo = PngInfo()
                        # Copy existing textual chunks (optional, be careful about duplicates)
                        for k, v in existing_info.items():
                            if isinstance(v, str):  # Only copy string chunks
                                # Avoid re-adding our own keyword key if it exists
                                if k.lower() != "keywords":
                                    pnginfo.add_text(k, v)

                        # Add combined tags as a new iTXt chunk (UTF-8)
                        keyword_key = "Keywords"  # A common key for tags
                        pnginfo.add_itxt(
                            keyword_key, tags_str, lang="en", tkey=keyword_key
                        )
                        print(f"  Prepared new PNG info with key '{keyword_key}'.")

                        # 3. Save the image with the new metadata
                        img.save(image_path, pnginfo=pnginfo)
                        write_success = True
                        print(
                            f"  Successfully added tags to PNG metadata chunk '{keyword_key}'."
                        )

                        # Update extracted_metadata with the key we added for consistency in logging
                        if "Info" not in extracted_metadata:
                            extracted_metadata["Info"] = {}
                        extracted_metadata["Info"][keyword_key] = tags_str

                    except Exception as png_e:
                        error_msg = f"Error processing PNG metadata: {png_e}"
                        print(f"  Error: {error_msg}")
                        # Ensure extracted_metadata is at least an empty dict if reading failed early
                        if extracted_metadata is None:
                            extracted_metadata = {}

                # --- Other Formats ---
                else:
                    error_msg = f"Unsupported format ({img_format}) for metadata writing. Only reading."
                    print(f"  {error_msg}")
                    # Try to read metadata if possible (e.g., img.info might exist)
                    try:
                        info_dict = img.info
                        if info_dict:
                            extracted_metadata = {"Info": info_dict.copy()}
                            print(
                                f"  Extracted metadata from Info dict for {img_format}."
                            )
                        else:
                            extracted_metadata = {}
                    except Exception as read_e:
                        print(
                            f"  Could not extract metadata for {img_format}: {read_e}"
                        )
                        extracted_metadata = {}

        # --- General Error Handling ---
        except UnidentifiedImageError:
            error_msg = "Cannot identify image file"
            print(f"  Error: {error_msg}")
        except FileNotFoundError:
            error_msg = "Image file not found during metadata processing"
            print(f"  Error: {error_msg}")
        except PermissionError:
            error_msg = "Permission denied for metadata processing"
            print(f"  Error: {error_msg}")
        except Exception as e:
            error_msg = f"Unexpected error processing metadata: {e}"
            print(f"  Error: {error_msg}")

        # Ensure extracted_metadata is a dict if it's still None after errors
        if extracted_metadata is None:
            extracted_metadata = {}

        # Return extracted data regardless of write success for logging purposes
        return extracted_metadata, write_success, error_msg

    def process_single_image(self, image_path: str, force_reprocess: bool = False):
        """Processes a single image: gets tags, extracts/adds metadata (incl. location), logs to DB."""
        basename = os.path.basename(image_path)
        abs_image_path = os.path.abspath(image_path)
        print(f"\nProcessing image: {basename} ({abs_image_path})")

        if not os.path.isfile(abs_image_path):
            print(f"Error: File not found: {abs_image_path}")
            # Log this specific error? Maybe not, as it didn't reach processing stages.
            return "File Not Found"

        # Check if skipped (only in batch mode, force=False)
        if not force_reprocess and self._is_already_processed(abs_image_path):
            print(
                "  Skipping: Already processed (found in DB) and force_reprocess=False."
            )
            return "Skipped - DB Record Exists"

        # 1. Get tags from Ollama
        ollama_tags, ollama_error = self._get_tags_from_ollama(abs_image_path)

        # Initialize final tags list
        final_tags = []
        if ollama_tags:
            final_tags.extend(ollama_tags)
        elif ollama_error:
            # Log Ollama error but proceed to metadata extraction if possible
            self._log_to_database(
                abs_image_path, None, None, "Ollama Error", ollama_error
            )
            # We might still want to extract existing metadata even if Ollama fails
            print(
                f"  Warning: Ollama failed ({ollama_error}), proceeding to metadata extraction only."
            )
            # Attempt metadata extraction without adding new tags
            extracted_metadata, _, metadata_error = self._extract_and_add_metadata(
                abs_image_path, []
            )  # Pass empty list
            # Update log with extracted metadata if any, keep Ollama error status
            self._log_to_database(
                abs_image_path, None, extracted_metadata, "Ollama Error", ollama_error
            )
            return "Ollama Error"  # Return original error status
        else:  # No tags and no specific error message from Ollama
            self._log_to_database(
                abs_image_path,
                None,
                None,
                "No Tags Received",
                "Ollama did not return any tags.",
            )
            # Proceed similar to Ollama error case
            print(
                "  Warning: Ollama returned no tags, proceeding to metadata extraction only."
            )
            extracted_metadata, _, metadata_error = self._extract_and_add_metadata(
                abs_image_path, []
            )
            self._log_to_database(
                abs_image_path,
                None,
                extracted_metadata,
                "No Tags Received",
                "Ollama did not return any tags.",
            )
            return "No Tags Received"

        # --- GPS Location Tagging ---
        location_tag = None
        gps_coords = self._get_gps_coordinates(abs_image_path)
        if gps_coords:
            lat, lon = gps_coords
            location_tag = self._get_location_from_coordinates(lat, lon)
            if location_tag:
                # Add location to the list if not already present from Ollama
                if location_tag not in final_tags:
                    final_tags.append(location_tag)
                print(f"  Added location tag: '{location_tag}'")
            else:
                print("  Could not determine location tag from GPS coordinates.")
                # Optionally log GPS presence even if geocoding failed?
        else:
            print("  No GPS coordinates found or readable.")
        # --- End GPS Location Tagging ---

        # 2. Extract existing metadata & Add combined tags to image metadata
        # Use the 'final_tags' list which includes Ollama tags + potentially location tag
        all_metadata, write_success, metadata_error = self._extract_and_add_metadata(
            abs_image_path, final_tags
        )

        # 3. Log result to database (always log, even if metadata write failed)
        if write_success:
            log_status = "Success"
            self._log_to_database(
                abs_image_path, final_tags, all_metadata, log_status, None
            )
            print(f"  Successfully processed and tagged: {basename}")
            return log_status
        else:
            # Log failure status, include extracted metadata if available
            log_status = "Metadata Error"
            # Log the final tags list even if writing failed, for debugging
            self._log_to_database(
                abs_image_path, final_tags, all_metadata, log_status, metadata_error
            )
            print(f"  Finished processing {basename} with metadata error.")
            return log_status

    def tag_images_in_folders(self):
        """Walks through the root directory and processes images."""
        if not self.root_folder:
            print("Error: Root folder not set. Cannot perform batch tagging.")
            return
        if not self.conn:
            print("Database connection not established. Aborting.")
            return

        print(f"\nStarting batch metadata tagging process in: {self.root_folder}")
        print("----------------------------------------")
        stats = {
            "found": 0,
            "success": 0,
            "skipped_db": 0,
            "ollama_error": 0,
            "no_tags": 0,
            "metadata_error": 0,
            "other_error": 0,
            "file_not_found": 0,  # Should be rare here
            "unsupported_format": 0,  # Track formats we don't write metadata for
        }
        start_time = time.time()

        # Get the absolute path and base name of the database file once
        db_abs_path = os.path.abspath(self.db_path)
        db_dir = os.path.dirname(db_abs_path)
        db_filename = os.path.basename(db_abs_path)

        for dirpath, _, filenames in os.walk(self.root_folder):
            current_dir_abs = os.path.abspath(dirpath)

            # Skip hidden directories (like .git, .vscode etc)
            # Check the directory name itself, not the full path
            if os.path.basename(current_dir_abs).startswith("."):
                print(f"\nSkipping hidden directory: {dirpath}")
                continue

            print(f"\nProcessing directory: {dirpath}")

            # Filter for supported image files initially
            image_files_in_dir = sorted(
                [f for f in filenames if f.lower().endswith(self.supported_extensions)]
            )

            # --- Correction: Filter out the DB file specifically ---
            # Check if the current directory is the one containing the database
            if current_dir_abs == db_dir:
                if db_filename in image_files_in_dir:
                    # This should not happen if DB has .db extension, but check anyway
                    print(f"  Ignoring database file found in list: {db_filename}")
                    image_files_in_dir.remove(db_filename)
                # Also remove the db file if it wasn't caught by extension filter
                elif db_filename in filenames:
                    print(f"  Ignoring database file: {db_filename}")
                    # No need to remove from image_files_in_dir as it wasn't added
                # --- End Correction ---

            if not image_files_in_dir:
                print("  No supported image files found (or only DB file was present).")
                continue

            num_files = len(image_files_in_dir)
            for i, filename in enumerate(image_files_in_dir):
                stats["found"] += 1
                image_path = os.path.join(
                    dirpath, filename
                )  # Use original dirpath here
                print(f"--- File {i + 1}/{num_files} ---")
                try:
                    # Use force_reprocess=False for batch mode (skip already processed)
                    result_status = self.process_single_image(
                        image_path, force_reprocess=False
                    )

                    # Update counters based on the returned status string
                    if result_status == "Success":
                        stats["success"] += 1
                    elif result_status == "Skipped - DB Record Exists":
                        stats["skipped_db"] += 1
                    elif result_status == "Ollama Error":
                        stats["ollama_error"] += 1
                    elif result_status == "No Tags Received":
                        stats["no_tags"] += 1
                    elif result_status == "Metadata Error":
                        stats["metadata_error"] += 1
                    elif result_status == "File Not Found":
                        stats["file_not_found"] += 1
                    else:
                        stats["other_error"] += 1
                        print(
                            f"Warning: Unknown status '{result_status}' for {filename}"
                        )

                except KeyboardInterrupt:
                    print("\n\nBatch processing interrupted by user. Exiting.")
                    self.close_db()
                    exit()
                except Exception as e:
                    stats["other_error"] += 1
                    print(f"!! Critical error processing {filename} in main loop: {e}")
                    try:
                        abs_path = os.path.abspath(image_path)
                        self._log_to_database(
                            abs_path, None, None, "Critical Error", str(e)
                        )
                    except Exception as log_e:
                        print(f"!! Failed to log critical error to DB: {log_e}")
                    time.sleep(1)

        end_time = time.time()
        duration = end_time - start_time
        print("\n----------------------------------------")
        print("Batch tagging process finished.")
        print(f"Total time: {duration:.2f} seconds")
        print("--- Statistics ---")
        print(f"Total images found:        {stats['found']}")
        print(f"Successfully processed:    {stats['success']}")
        print(f"Skipped (already in DB): {stats['skipped_db']}")
        print(f"Ollama errors:           {stats['ollama_error']}")
        print(f"Ollama - no tags:        {stats['no_tags']}")
        print(f"Metadata write errors:   {stats['metadata_error']}")
        print(f"File not found errors:   {stats['file_not_found']}")
        print(f"Other critical errors:   {stats['other_error']}")
        print("----------------------------------------")
        print(f"Results logged to: {self.db_path}")

    # In c:\Users\frank\Documents\py_projects\dev\image_tag\img_tagger.py
    # Inside the ImageMetadataManager class

    def search_images(
        self, tags: list[str] | None = None, location_keyword: str | None = None
    ):
        """Searches the database for images matching tags or location keywords."""
        if not self.cursor:
            print("Error: Database connection not available for searching.")
            return

        query = (
            "SELECT original_path, ollama_tags, status FROM processed_images WHERE 1=1"
        )
        params = []
        conditions = []  # Store individual AND conditions
        search_terms = []  # Keep track of search terms for display

        # --- Tag Search Logic (remains the same) ---
        if tags:
            tag_clauses = []
            search_terms.extend([f"tag: {t}" for t in tags])
            for tag in tags:
                # Search for the tag as a distinct JSON element "tag"
                tag_clauses.append("ollama_tags LIKE ?")
                params.append(f"%{tag.lower().strip()}%")
            if tag_clauses:
                # Find images with ANY of the tags (OR logic within this group)
                conditions.append("(" + " OR ".join(tag_clauses) + ")")

        # --- Location Search Logic (Corrected for CLI) ---
        if location_keyword:
            loc_lower = location_keyword.lower().strip()
            search_terms.append(f"location: {loc_lower}")
            print(f"Searching for location keyword '{loc_lower}' in stored tags...")
            # Search for the location keyword as a substring anywhere in the tags field
            conditions.append("ollama_tags LIKE ?")
            params.append(f"%{loc_lower}%")

        # --- Combine conditions and execute ---
        if not search_terms:
            # Check moved here to allow combining tag and location search terms display
            print("Error: No search criteria provided (use --tags or --location).")
            return

        if conditions:
            query += " AND " + " AND ".join(conditions)

        print(f"\nSearching for images with criteria: {', '.join(search_terms)}")

        try:
            self.cursor.execute(query, params)
            results = self.cursor.fetchall()

            if not results:
                print("No matching images found in the database.")
                return

            print(f"\nFound {len(results)} matching images:")
            print("-" * 40)
            for row in results:
                path, tags_json, status = row
                try:
                    tags_list = json.loads(tags_json) if tags_json else []
                except json.JSONDecodeError:
                    tags_list = ["Error decoding tags"]

                print(f"Path: {path}")
                print(f"  Tags: {', '.join(tags_list)}")
                print(f"  Status: {status}")
                print("-" * 20)

        except sqlite3.Error as e:
            print(f"Error searching database: {e}")
            # Hint about old SQLite remains useful
            if "no such function: json_extract" in str(
                e
            ) or "no such function: json_each" in str(e):
                print(
                    "Hint: Your SQLite version might be too old for JSON functions. The current LIKE search should work, but this message indicates potential limitations."
                )

    def view_record(self, image_path: str):
        """Displays the full database record for a specific image."""
        if not self.cursor:
            print("Error: Database connection not available.")
            return
        abs_path = os.path.abspath(image_path)
        try:
            self.cursor.execute(
                "SELECT * FROM processed_images WHERE original_path = ?", (abs_path,)
            )
            record = self.cursor.fetchone()
            if record:
                print("\nDatabase Record:")
                print("-" * 30)
                col_names = [desc[0] for desc in self.cursor.description]
                record_dict = dict(zip(col_names, record))

                for key, value in record_dict.items():
                    col_title = key.replace("_", " ").title()
                    if key in ["ollama_tags", "all_exif_data"] and value:
                        try:
                            # Pretty print JSON fields
                            parsed_json = json.loads(value)
                            print(f"{col_title}:")
                            # Use ensure_ascii=False for potentially non-ASCII chars in metadata
                            print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
                        except json.JSONDecodeError:
                            print(f"{col_title}: (Invalid JSON in DB)")
                            print(value)  # Print raw value if JSON is broken
                    elif value is None:
                        print(f"{col_title}: None")
                    else:
                        print(f"{col_title}: {value}")
                print("-" * 30)
            else:
                print(f"No record found for path: {abs_path}")
        except sqlite3.Error as e:
            print(f"Error viewing record for {abs_path}: {e}")

    def delete_record(self, image_path: str):
        """Deletes the database record for a specific image."""
        if not self.cursor or not self.conn:
            print("Error: Database connection not available.")
            return
        abs_path = os.path.abspath(image_path)
        try:
            # Check if record exists first
            self.cursor.execute(
                "SELECT 1 FROM processed_images WHERE original_path = ?", (abs_path,)
            )
            if self.cursor.fetchone():
                self.cursor.execute(
                    "DELETE FROM processed_images WHERE original_path = ?", (abs_path,)
                )
                # Since isolation_level=None, changes are committed automatically.
                # Verify deletion (optional)
                self.cursor.execute("SELECT changes()")
                changes = self.cursor.fetchone()[0]
                if changes > 0:
                    print(f"Successfully deleted database record for: {abs_path}")
                else:
                    print(f"Attempted delete, but no rows affected for: {abs_path}")

            else:
                print(f"No record found to delete for path: {abs_path}")
        except sqlite3.Error as e:
            print(f"Error deleting record for {abs_path}: {e}")

    def update_record_tags(self, image_path: str, new_tags: list[str]):
        """Updates only the ollama_tags field in the database for a specific image."""
        # NOTE: This does NOT update the metadata in the image file itself.
        if not self.cursor or not self.conn:
            print("Error: Database connection not available.")
            return
        abs_path = os.path.abspath(image_path)
        # Sanitize and prepare tags
        clean_tags = sorted(list(set([t.lower().strip() for t in new_tags])))
        tags_json = json.dumps(clean_tags)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            # Check if record exists first
            self.cursor.execute(
                "SELECT 1 FROM processed_images WHERE original_path = ?", (abs_path,)
            )
            if self.cursor.fetchone():
                self.cursor.execute(
                    """
                    UPDATE processed_images
                    SET ollama_tags = ?, processed_timestamp = ?, status = 'Manually Updated', error_message = NULL
                    WHERE original_path = ?
                """,
                    (tags_json, timestamp, abs_path),
                )
                # Verify update (optional)
                self.cursor.execute("SELECT changes()")
                changes = self.cursor.fetchone()[0]
                if changes > 0:
                    print(f"Successfully updated database tags for: {abs_path}")
                    print(f"  New tags: {', '.join(clean_tags)}")
                    print("  Note: Image file metadata was NOT modified.")
                else:
                    print(
                        f"Attempted update, but no rows affected for: {abs_path} (maybe tags were identical?)"
                    )

            else:
                print(f"No record found to update for path: {abs_path}")
        except sqlite3.Error as e:
            print(f"Error updating record tags for {abs_path}: {e}")

    def close_db(self):
        """Closes the database connection."""
        if self.conn:
            db_path = self.db_path  # Store path before closing
            try:
                # Optional: Optimize DB before closing in WAL mode
                print("Optimizing database...")
                self.conn.execute("PRAGMA optimize;")
                # Optional: Checkpoint WAL file before closing
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                self.conn.close()
                self.conn = None  # Prevent further use
                self.cursor = None
                print(f"Database connection closed ({db_path}).")
            except sqlite3.Error as e:
                print(f"Error during database closing/optimization: {e}")


# --- Main Execution & CLI Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tag images using Ollama and GPS data, store metadata in SQLite, and manage records."
    )
    parser.add_argument(
        "--db-path",
        help="Optional: Specify a path for the database file. Overrides default locations.",
        default=None,  # Explicitly default to None
    )

    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    # --- Tag All Command ---
    parser_tag_all = subparsers.add_parser(
        "tag-all", help="Recursively find and tag all images in a directory."
    )
    parser_tag_all.add_argument(
        "root_folder", help="The root directory containing images to process."
    )
    # Note: --db-path is now a top-level argument

    # --- Tag Single Command ---
    parser_tag_single = subparsers.add_parser(
        "tag-single", help="Tag a single image file."
    )
    parser_tag_single.add_argument(
        "image_path", help="Path to the single image file to process."
    )
    parser_tag_single.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if already in database.",
    )
    # Note: --db-path is now a top-level argument

    # --- Search Command ---
    parser_search = subparsers.add_parser(
        "search", help="Search the database for images."
    )
    parser_search.add_argument(
        "--tags", nargs="+", help="One or more tags to search for (matches any)."
    )
    parser_search.add_argument(
        "--location",
        help="Keyword to search for in stored tags (e.g., city name from GPS).",
    )
    # Note: --db-path is now a top-level argument, but marked required conceptually for this command

    # --- View Command ---
    parser_view = subparsers.add_parser(
        "view", help="View the database record for a specific image."
    )
    parser_view.add_argument(
        "image_path", help="Path to the image file whose record you want to view."
    )
    # Note: --db-path is now a top-level argument, but marked required conceptually

    # --- Delete Command ---
    parser_delete = subparsers.add_parser(
        "delete", help="Delete the database record for a specific image."
    )
    parser_delete.add_argument(
        "image_path", help="Path to the image file whose record you want to delete."
    )
    # Note: --db-path is now a top-level argument, but marked required conceptually

    # --- Update Command (DB Only) ---
    parser_update = subparsers.add_parser(
        "update-tags",
        help="Update the Ollama tags in the DB record (does NOT modify image file).",
    )
    parser_update.add_argument(
        "image_path", help="Path to the image file whose record you want to update."
    )
    parser_update.add_argument(
        "--tags", nargs="+", required=True, help="The new list of tags."
    )
    # Note: --db-path is now a top-level argument, but marked required conceptually

    args = parser.parse_args()

    manager = None
    db_path_to_use = args.db_path  # Use the top-level argument if provided

    try:
        # Initialize Manager based on command and db_path
        if args.command == "tag-all":
            # For tag-all, root_folder determines default DB path if --db-path is not given
            manager = ImageMetadataManager(
                root_folder=args.root_folder, db_path=db_path_to_use
            )
            manager.tag_images_in_folders()
        elif args.command == "tag-single":
            # For single, db_path defaults to CWD if --db-path is not given
            manager = ImageMetadataManager(db_path=db_path_to_use)
            manager.process_single_image(args.image_path, force_reprocess=args.force)
        else:
            # For DB management commands (search, view, delete, update), db_path is needed.
            # The __init__ handles defaulting to CWD if db_path_to_use is None,
            # but these commands are less useful without a specific DB.
            # We'll rely on the manager's initialization logic.
            if not db_path_to_use:
                # If no specific DB path is given for these commands, try CWD default.
                # Consider making --db-path required for these subcommands if CWD default is undesirable.
                print(
                    "Warning: No --db-path specified. Using default database location (likely current working directory)."
                )

            manager = ImageMetadataManager(db_path=db_path_to_use)

            if args.command == "search":
                if not args.tags and not args.location:
                    print("Error: Please provide --tags or --location for searching.")
                    parser_search.print_help()
                else:
                    manager.search_images(
                        tags=args.tags, location_keyword=args.location
                    )
            elif args.command == "view":
                manager.view_record(args.image_path)
            elif args.command == "delete":
                manager.delete_record(args.image_path)
            elif args.command == "update-tags":
                manager.update_record_tags(args.image_path, args.tags)

        print("\nOperation finished.")

    except ValueError as ve:
        print(f"Configuration Error: {ve}")
    except ConnectionError as ce:
        print(f"Database Connection Error: {ce}")
    except FileNotFoundError as fnf:
        print(f"Error: File not found - {fnf}")
    except PermissionError as pe:
        print(f"Error: Permission denied - {pe}")
    except ImportError as ie:
        if "exifread" in str(ie) or "geopy" in str(ie):
            print(f"Error: Missing required library. Please install it using:")
            print(f"  pip install exifread geopy")
        else:
            print(f"An unexpected import error occurred: {ie}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Uncomment for detailed debugging
        # import traceback
        # traceback.print_exc()
    finally:
        if manager:
            manager.close_db()
