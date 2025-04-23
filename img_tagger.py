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
            exif_with_names[ifd_name][tag_name] = value  # Keep original value for now
    return exif_with_names


class ImageMetadataManager:
    """
    Manages image metadata tagging, storage, and retrieval using Ollama and SQLite.
    Includes CLI interaction logic.
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
            ".png",
            ".gif",
            ".bmp",
            ".tiff",
            ".webp",
        )
        self.conn = None
        self.cursor = None
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
                    ollama_tags TEXT,       -- JSON list of tags from Ollama
                    all_exif_data TEXT,     -- JSON blob of all extracted EXIF
                    processed_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    status TEXT,            -- e.g., 'Success', 'No Tags', 'Metadata Error', 'Ollama Error'
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
        all_exif: dict | None,
        status: str,
        error_msg: str | None = None,
    ):
        """Logs or updates the processing result in the database."""
        if not self.conn or not self.cursor:
            print("Error: Database connection not available for logging.")
            return

        tags_json = json.dumps(ollama_tags) if ollama_tags else None
        # Sanitize EXIF data before converting to JSON
        exif_json = json.dumps(_sanitize_for_json(all_exif)) if all_exif else None
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
                (original_path, tags_json, exif_json, timestamp, status, error_msg),
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
        print(f"  Attempting to get tags for: {os.path.basename(image_path)}...")
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
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
            response.raise_for_status()
            response_data = response.json()
            raw_tags_text = response_data.get("response", "").strip()

            if not raw_tags_text:
                return None, "Ollama returned empty response"

            tags = [
                tag.strip().lower() for tag in raw_tags_text.split(",") if tag.strip()
            ]
            tags = list(dict.fromkeys(tags))  # Remove duplicates

            print(f"  Tags received: {tags}")
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

    def _extract_and_add_metadata(
        self, image_path: str, tags: list[str]
    ) -> tuple[dict | None, bool, str | None]:
        """
        Extracts existing metadata, adds Ollama tags (to UserComment for EXIF,
        or a text chunk for PNG), and returns all extracted metadata.

        Returns:
            A tuple: (all_extracted_metadata_dict or None, success_writing_tags boolean, error_message string or None)
        """
        print(f"  Processing metadata for: {os.path.basename(image_path)}")
        tags_str = ", ".join(tags)
        extracted_metadata = None  # Changed variable name for clarity
        write_success = False
        error_msg = None

        try:
            # Use 'with' statement for reliable file handling with Pillow
            with Image.open(image_path) as img:
                img_format = img.format
                # --- JPEG/TIFF Handling (Existing Logic) ---
                if img_format in ["JPEG", "TIFF"]:
                    try:
                        exif_dict = piexif.load(
                            img.info.get("exif", b"")
                        )  # Load from info if available
                        # Get human-readable names for storage/viewing
                        extracted_metadata = _get_exif_with_names(exif_dict)
                    except (piexif.InvalidImageDataError, ValueError, KeyError) as e:
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
                        extracted_metadata = {}

                    try:
                        if "Exif" not in exif_dict:
                            exif_dict["Exif"] = {}
                        exif_dict["Exif"][piexif.ExifIFD.UserComment] = (
                            piexif.helper.UserComment.dump(tags_str, encoding="unicode")
                        )
                        exif_bytes = piexif.dump(exif_dict)
                        # Save back using Pillow's save method with exif bytes
                        img.save(image_path, exif=exif_bytes)  # Use Pillow's save
                        print("  Successfully added tags to EXIF UserComment.")
                        write_success = True
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
                        # Optionally copy existing textual chunks (be careful not to duplicate)
                        # for k, v in existing_info.items():
                        #     if isinstance(v, str):
                        #         pnginfo.add_text(k, v) # Copies existing text

                        # Add Ollama tags as a new iTXt chunk (UTF-8)
                        keyword_key = "Keywords"  # A common key for tags
                        pnginfo.add_itxt(
                            keyword_key, tags_str, lang="en", tkey=keyword_key
                        )
                        print(f"  Prepared new PNG info with key '{keyword_key}'.")

                        # 3. Save the image with the new metadata
                        # NOTE: This re-saves the entire PNG file.
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
        """Processes a single image: gets tags, extracts/adds metadata, logs to DB."""
        basename = os.path.basename(image_path)
        abs_image_path = os.path.abspath(image_path)
        print(f"Processing image: {basename}")

        if not os.path.isfile(abs_image_path):
            print(f"Error: File not found: {abs_image_path}")
            # Log this specific error? Maybe not, as it didn't reach processing stages.
            return "File Not Found"

        # For single file processing, 'force_reprocess' controls overwriting.
        # The DB logging uses INSERT OR REPLACE, so it always updates/inserts.
        if not force_reprocess and self._is_already_processed(abs_image_path):
            # This check might be redundant if we always want tag-single to reprocess.
            # Keep it for now, controlled by the flag.
            print("  Skipping: Already processed and force_reprocess=False.")
            return "Skipped - DB Record Exists"

        # 1. Get tags from Ollama
        ollama_tags, ollama_error = self._get_tags_from_ollama(abs_image_path)

        if ollama_error or not ollama_tags:
            log_status = "Ollama Error" if ollama_error else "No Tags Received"
            log_msg = ollama_error or "Ollama did not return any tags."
            # Log failure, store no EXIF data as we didn't get tags to write
            self._log_to_database(abs_image_path, None, None, log_status, log_msg)
            return log_status

        # 2. Extract existing metadata & Add Ollama tags to image metadata
        all_exif, write_success, metadata_error = self._extract_and_add_metadata(
            abs_image_path, ollama_tags
        )

        # 3. Log result to database (always log, even if metadata write failed)
        if write_success:
            log_status = "Success"
            self._log_to_database(
                abs_image_path, ollama_tags, all_exif, log_status, None
            )
            return log_status
        else:
            # Log failure status, include extracted EXIF if available
            log_status = "Metadata Error"
            self._log_to_database(
                abs_image_path, ollama_tags, all_exif, log_status, metadata_error
            )
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
            "file_not_found": 0,
        }

        for dirpath, _, filenames in os.walk(self.root_folder):
            # Skip the directory where the database itself is located
            if os.path.abspath(dirpath) == os.path.dirname(self.db_path):
                print(f"\nSkipping database directory: {dirpath}")
                continue

            print(f"\nProcessing directory: {dirpath}")
            image_files_in_dir = [
                f for f in filenames if f.lower().endswith(self.supported_extensions)
            ]

            if not image_files_in_dir:
                print("  No supported image files found.")
                continue

            for filename in image_files_in_dir:
                stats["found"] += 1
                image_path = os.path.join(dirpath, filename)
                try:
                    # Use force_reprocess=False for batch mode (skip already processed)
                    result_status = self.process_single_image(
                        image_path, force_reprocess=False
                    )

                    # Update counters
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
                        stats["file_not_found"] += 1  # Should be rare here
                    else:
                        stats["other_error"] += 1
                        print(
                            f"Warning: Unknown status '{result_status}' for {filename}"
                        )

                except Exception as e:
                    stats["other_error"] += 1
                    print(f"!! Critical error processing {filename} in main loop: {e}")
                    self._log_to_database(
                        image_path, None, None, "Critical Error", str(e)
                    )
                    time.sleep(1)

        print("\n----------------------------------------")
        print("Batch tagging process finished.")
        # Print stats... (omitted for brevity, same as before)
        print(f"Results logged to: {self.db_path}")

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

        # --- Tag Search ---
        # Simple search: finds images where *any* of the provided tags are present.
        # Uses JSON_EXTRACT (requires recent SQLite) or LIKE for broader compatibility.
        # For more complex logic (e.g., all tags must match), adjust the query.
        if tags:
            tag_clauses = []
            for tag in tags:
                # Option 1: Using json_each (potentially more efficient if indexed)
                # query += f" AND id IN (SELECT id FROM processed_images, json_each(ollama_tags) WHERE json_each.value = ?)"
                # params.append(tag.lower())

                # Option 2: Using LIKE (more compatible, might be slower)
                tag_clauses.append("ollama_tags LIKE ?")
                params.append(
                    f'%"{tag.lower()}"%'
                )  # Search for the tag within the JSON array string

            if tag_clauses:
                query += (
                    " AND (" + " OR ".join(tag_clauses) + ")"
                )  # Find images with ANY of the tags

        # --- Location Search (Basic Keyword in Tags) ---
        # TODO: Enhance this to parse GPS data from all_exif_data if needed
        if location_keyword:
            print(
                f"Searching for location keyword '{location_keyword}' in Ollama tags..."
            )
            query += " AND ollama_tags LIKE ?"
            params.append(
                f'%"{location_keyword.lower()}"%'
            )  # Search within the JSON array string

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
                tags_list = json.loads(tags_json) if tags_json else []
                print(f"Path: {path}")
                print(f"  Tags: {', '.join(tags_list)}")
                print(f"  Status: {status}")
                print("-" * 20)

        except sqlite3.Error as e:
            print(f"Error searching database: {e}")
            if "no such function: json_extract" in str(
                e
            ) or "no such function: json_each" in str(e):
                print(
                    "Hint: Your SQLite version might be too old for JSON functions. Consider using the LIKE search method."
                )

    def view_record(self, image_path: str):
        """Displays the full database record for a specific image."""
        if not self.cursor:
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
                # Assuming column order: id, path, ollama_tags, all_exif, timestamp, status, error
                col_names = [desc[0] for desc in self.cursor.description]
                record_dict = dict(zip(col_names, record))

                for key, value in record_dict.items():
                    if key in ["ollama_tags", "all_exif_data"] and value:
                        try:
                            # Pretty print JSON fields
                            parsed_json = json.loads(value)
                            print(f"{key.replace('_', ' ').title()}:")
                            print(json.dumps(parsed_json, indent=2))
                        except json.JSONDecodeError:
                            print(
                                f"{key.replace('_', ' ').title()}: (Invalid JSON in DB)"
                            )
                            print(value)
                    else:
                        print(f"{key.replace('_', ' ').title()}: {value}")
                print("-" * 30)
            else:
                print(f"No record found for path: {abs_path}")
        except sqlite3.Error as e:
            print(f"Error viewing record for {abs_path}: {e}")

    def delete_record(self, image_path: str):
        """Deletes the database record for a specific image."""
        if not self.cursor:
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
                print(f"Successfully deleted database record for: {abs_path}")
            else:
                print(f"No record found to delete for path: {abs_path}")
        except sqlite3.Error as e:
            print(f"Error deleting record for {abs_path}: {e}")

    def update_record_tags(self, image_path: str, new_tags: list[str]):
        """Updates only the ollama_tags field in the database for a specific image."""
        # NOTE: This does NOT update the metadata in the image file itself.
        if not self.cursor:
            return
        abs_path = os.path.abspath(image_path)
        tags_json = json.dumps(new_tags)
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
                print(f"Successfully updated database tags for: {abs_path}")
                print("Note: Image file metadata was NOT modified.")
            else:
                print(f"No record found to update for path: {abs_path}")
        except sqlite3.Error as e:
            print(f"Error updating record tags for {abs_path}: {e}")

    def close_db(self):
        """Closes the database connection."""
        if self.conn:
            try:
                self.conn.close()
                print("Database connection closed.")
            except sqlite3.Error as e:
                print(f"Error closing database connection: {e}")


# --- Main Execution & CLI Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tag images using Ollama, store metadata in SQLite, and manage records."
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
    # parser_tag_all.add_argument("--db-path", help="Optional: Specify a custom path for the database file.") # Add if needed

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
    parser_tag_single.add_argument(
        "--db-path",
        help="Optional: Specify the path to the database file to use/create.",
    )

    # --- Search Command ---
    parser_search = subparsers.add_parser(
        "search", help="Search the database for images."
    )
    parser_search.add_argument(
        "--tags", nargs="+", help="One or more tags to search for (matches any)."
    )
    parser_search.add_argument(
        "--location",
        help="Keyword to search for in Ollama tags (basic location search).",
    )
    parser_search.add_argument(
        "--db-path", required=True, help="Path to the database file to search."
    )  # Require DB for search

    # --- View Command ---
    parser_view = subparsers.add_parser(
        "view", help="View the database record for a specific image."
    )
    parser_view.add_argument(
        "image_path", help="Path to the image file whose record you want to view."
    )
    parser_view.add_argument(
        "--db-path", required=True, help="Path to the database file."
    )

    # --- Delete Command ---
    parser_delete = subparsers.add_parser(
        "delete", help="Delete the database record for a specific image."
    )
    parser_delete.add_argument(
        "image_path", help="Path to the image file whose record you want to delete."
    )
    parser_delete.add_argument(
        "--db-path", required=True, help="Path to the database file."
    )

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
    parser_update.add_argument(
        "--db-path", required=True, help="Path to the database file."
    )

    args = parser.parse_args()

    manager = None
    try:
        # Initialize Manager based on command
        if args.command == "tag-all":
            # db_path = args.db_path if hasattr(args, 'db_path') else None # Get optional db path if added
            manager = ImageMetadataManager(
                root_folder=args.root_folder
            )  # DB path derived from root
            manager.tag_images_in_folders()
        elif args.command == "tag-single":
            # For single, DB path can be specified or defaults to CWD
            manager = ImageMetadataManager(db_path=args.db_path)
            manager.process_single_image(args.image_path, force_reprocess=args.force)
        elif args.command == "search":
            manager = ImageMetadataManager(db_path=args.db_path)  # Only need DB path
            if not args.tags and not args.location:
                print("Error: Please provide --tags or --location for searching.")
            else:
                manager.search_images(tags=args.tags, location_keyword=args.location)
        elif args.command == "view":
            manager = ImageMetadataManager(db_path=args.db_path)
            manager.view_record(args.image_path)
        elif args.command == "delete":
            manager = ImageMetadataManager(db_path=args.db_path)
            manager.delete_record(args.image_path)
        elif args.command == "update-tags":
            manager = ImageMetadataManager(db_path=args.db_path)
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
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Consider adding more specific exception handling or logging traceback for debugging
        # import traceback
        # traceback.print_exc()
    finally:
        if manager:
            manager.close_db()
