import gradio as gr
import os
import sqlite3
import json
import pandas as pd  # For better display of search results
from datetime import datetime

# --- Import necessary components from your script ---
# Assuming img_tagger.py is in the same directory or accessible via PYTHONPATH
try:
    # If img_tagger.py is in the same directory:
    from img_tagger import (
        ImageMetadataManager,
        _sanitize_for_json,
    )  # Import the class and helper

    print("Successfully imported ImageMetadataManager from img_tagger.py")
except ImportError:
    print("Error: Could not import ImageMetadataManager.")
    print("Make sure 'img_tagger.py' is in the same directory or your PYTHONPATH.")

    # Define a dummy class if import fails, so Gradio doesn't crash immediately
    class ImageMetadataManager:
        def __init__(self, *args, **kwargs):
            raise ImportError("ImageMetadataManager could not be loaded.")

        def search_images(*args, **kwargs):
            return (
                [],
                pd.DataFrame(),
                "Error: Manager not loaded.",
            )  # Added DataFrame for consistency

        def view_record(*args, **kwargs):
            return None, "Error: Manager not loaded."

        def update_record_tags(*args, **kwargs):
            return "Error: Manager not loaded."

        def delete_record(*args, **kwargs):
            return "Error: Manager not loaded."

        def process_single_image(*args, **kwargs):
            return "Error: Manager not loaded.", None  # Added None for image output

        def close_db(self):
            pass

    # Exit if the core class isn't available
    # import sys
    # sys.exit("Exiting due to import error.")
    # Or, let Gradio show the error when actions are attempted.


# --- Gradio Wrapper Functions ---


# Helper to instantiate the manager safely
def get_manager(db_path):
    if not db_path:
        raise gr.Error("Database path is not set. Please enter a valid path.")
    # Check if the directory exists, not the file itself, as it might be created
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(
        db_dir
    ):  # Handle cases where db_path is just a filename
        raise gr.Error(f"The directory for the database does not exist: {db_dir}")
    # Also check if the file itself exists for read operations
    if not os.path.exists(db_path):
        raise gr.Error(f"Database file not found at: {db_path}")
    try:
        # Instantiate the manager for each operation to ensure clean state/connection handling
        # Pass only db_path, as root_folder isn't needed for these UI operations
        manager = ImageMetadataManager(db_path=db_path)
        return manager
    except (
        ValueError,
        ConnectionError,
        ImportError,
        sqlite3.Error,
    ) as e:  # Catch sqlite3 errors too
        raise gr.Error(
            f"Failed to initialize ImageMetadataManager or connect to DB: {e}"
        )
    except Exception as e:
        raise gr.Error(f"An unexpected error occurred initializing manager: {e}")


# --- Search Function Wrapper ---
def search_db(db_path, search_tags_str, location_keyword):
    """Wraps manager.search_images for Gradio, returning structured data."""
    manager = None
    empty_df = pd.DataFrame(columns=["Path", "Tags", "Status", "Timestamp", "Exists"])
    try:
        manager = get_manager(db_path)
        tags_list = (
            [tag.strip().lower() for tag in search_tags_str.split(",") if tag.strip()]
            if search_tags_str
            else None
        )
        loc_keyword = location_keyword.strip().lower() if location_keyword else None

        if not tags_list and not loc_keyword:
            return (
                [],
                empty_df,
                "Please enter tags or a location keyword to search.",
            )

        # Query directly for better control in the UI wrapper.
        query = "SELECT original_path, ollama_tags, status, processed_timestamp FROM processed_images WHERE 1=1"
        params = []
        conditions = []  # Store individual AND conditions

        # --- Tag Search Logic (remains the same) ---
        if tags_list:
            tag_clauses = []
            for tag in tags_list:
                # Search for the tag as a distinct JSON element "tag"
                tag_clauses.append("ollama_tags LIKE ?")
                params.append(f"%{tag}%")
            if tag_clauses:
                # Find images with ANY of the tags (OR logic within this group)
                conditions.append("(" + " OR ".join(tag_clauses) + ")")

        # --- Location Search Logic (Corrected) ---
        if loc_keyword:
            # Search for the location keyword as a substring anywhere in the tags field
            conditions.append("ollama_tags LIKE ?")
            params.append(
                f"%{loc_keyword}%"
            )  # <<< REMOVED the double quotes around the keyword

        # Combine all conditions with AND
        if conditions:
            query += " AND " + " AND ".join(conditions)

        # --- Execute Query ---
        manager.cursor.execute(query, params)
        results = manager.cursor.fetchall()  # List of tuples

        if not results:
            return (
                [],
                empty_df,
                "No matching images found.",
            )

        # --- Process Results (remains the same) ---
        image_paths = []
        data_for_df = []
        found_count = len(results)
        display_count = 0

        for row in results:
            path, tags_json, status, timestamp = row
            tags_list_from_db = []
            tags_str_display = ""
            if tags_json:
                try:
                    tags_list_from_db = json.loads(tags_json)
                    tags_str_display = ", ".join(tags_list_from_db)
                except json.JSONDecodeError:
                    tags_str_display = "(Invalid JSON in DB)"

            file_exists = os.path.exists(path)
            # Check if image file exists before adding to gallery
            if file_exists:
                image_paths.append(path)
                display_count += 1

            data_for_df.append(
                {
                    "Path": path,
                    "Tags": tags_str_display,
                    "Status": status,
                    "Timestamp": timestamp,
                    "Exists": file_exists,  # Add existence check to table
                }
            )

        df = pd.DataFrame(data_for_df)
        status_msg = f"Found {found_count} record(s)."
        if found_count > display_count:
            status_msg += f" Displaying {display_count} existing image(s) in gallery."
            status_msg += (
                f" ({found_count - display_count} image paths not found on disk)."
            )
        else:
            status_msg += f" Displaying all {display_count} in gallery."

        return image_paths, df, status_msg

    except gr.Error as e:  # Catch Gradio errors from get_manager
        return [], empty_df, str(e)
    except sqlite3.Error as e:
        return [], empty_df, f"Database search error: {e}"
    except Exception as e:
        return [], empty_df, f"An unexpected error occurred during search: {e}"
    finally:
        if manager:
            manager.close_db()


# --- View Record Function Wrapper ---
def view_db_record(db_path, image_path):
    """Wraps manager.view_record for Gradio."""
    manager = None
    if not image_path:
        return (
            None,
            "Please enter an image path to view or select one from the gallery.",
        )
    abs_path = os.path.abspath(image_path)
    try:
        manager = get_manager(db_path)
        # Query directly to get structured data easily
        manager.cursor.execute(
            "SELECT * FROM processed_images WHERE original_path = ?", (abs_path,)
        )
        record = manager.cursor.fetchone()

        if record:
            col_names = [desc[0] for desc in manager.cursor.description]
            record_dict = dict(zip(col_names, record))

            # Attempt to parse JSON fields for better display
            for key in ["ollama_tags", "all_exif_data"]:
                if key in record_dict and isinstance(record_dict[key], str):
                    try:
                        record_dict[key] = json.loads(record_dict[key])
                    except json.JSONDecodeError:
                        record_dict[key] = f"(Invalid JSON in DB: {record_dict[key]})"

            # Sanitize the whole dict just in case before returning as JSON object
            # (May not be strictly necessary for gr.JSON, but good practice)
            sanitized_record = _sanitize_for_json(record_dict)
            return sanitized_record, f"Record found for {os.path.basename(abs_path)}"
        else:
            return None, f"No record found for path: {abs_path}"

    except gr.Error as e:
        return None, str(e)
    except sqlite3.Error as e:
        return None, f"Database view error: {e}"
    except Exception as e:
        return None, f"An unexpected error occurred viewing record: {e}"
    finally:
        if manager:
            manager.close_db()


# --- Update Tags (DB Only) Function Wrapper ---
def update_db_tags(db_path, image_path, new_tags_str):
    """Wraps manager.update_record_tags for Gradio."""
    manager = None
    if not image_path:
        return "Please provide the image path (e.g., select from gallery)."
    if not new_tags_str:
        return "Please provide the new tags."

    abs_path = os.path.abspath(image_path)
    new_tags_list = sorted(
        [tag.strip().lower() for tag in new_tags_str.split(",") if tag.strip()]
    )
    if not new_tags_list:
        return "Please provide valid tags (non-empty after stripping)."

    try:
        manager = get_manager(db_path)
        # Replicate the core logic here to return the status directly.
        tags_json = json.dumps(new_tags_list)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        manager.cursor.execute(
            "SELECT 1 FROM processed_images WHERE original_path = ?", (abs_path,)
        )
        if manager.cursor.fetchone():
            manager.cursor.execute(
                """
                UPDATE processed_images
                SET ollama_tags = ?, processed_timestamp = ?, status = 'Manually Updated', error_message = NULL
                WHERE original_path = ?
            """,
                (tags_json, timestamp, abs_path),
            )
            # manager.conn.commit() # Not needed if autocommit (isolation_level=None)
            return f"Successfully updated database tags for: {os.path.basename(abs_path)}. New tags: {', '.join(new_tags_list)}"
        else:
            return f"No record found to update for path: {abs_path}"

    except gr.Error as e:
        return str(e)
    except sqlite3.Error as e:
        return f"Database update error: {e}"
    except Exception as e:
        return f"An unexpected error occurred updating tags: {e}"
    finally:
        if manager:
            manager.close_db()


# --- Delete Record Function Wrapper ---
def delete_db_record(db_path, image_path):
    """Wraps manager.delete_record for Gradio."""
    manager = None
    if not image_path:
        return "Please provide the image path to delete (e.g., select from gallery)."
    abs_path = os.path.abspath(image_path)
    try:
        manager = get_manager(db_path)
        # Replicate logic to return status
        manager.cursor.execute(
            "SELECT 1 FROM processed_images WHERE original_path = ?", (abs_path,)
        )
        if manager.cursor.fetchone():
            manager.cursor.execute(
                "DELETE FROM processed_images WHERE original_path = ?", (abs_path,)
            )
            # manager.conn.commit() # Not needed if autocommit
            return f"Successfully deleted database record for: {os.path.basename(abs_path)}"
        else:
            return f"No record found to delete for path: {abs_path}"

    except gr.Error as e:
        return str(e)
    except sqlite3.Error as e:
        return f"Database delete error: {e}"
    except Exception as e:
        return f"An unexpected error occurred deleting record: {e}"
    finally:
        if manager:
            manager.close_db()


# --- Tag Single Image Function Wrapper ---
def tag_single_image_ui(db_path, image_path, force_reprocess):
    """Wraps manager.process_single_image for Gradio."""
    manager = None
    if not image_path:
        return (
            "Please provide the image path to process (e.g., select from gallery).",
            None,
        )  # Status, Output Image
    abs_path = os.path.abspath(image_path)
    if not os.path.exists(abs_path):
        return f"File not found: {abs_path}", None

    # Display the image being processed
    output_image_display = abs_path

    try:
        # Instantiate with db_path only, process_single_image uses the absolute path
        # Need to handle potential errors during manager init here too
        manager = ImageMetadataManager(
            db_path=db_path
        )  # Instantiate here to handle creation if needed
        # This function modifies the image file and updates the DB
        # It returns a status string.
        status = manager.process_single_image(abs_path, force_reprocess=force_reprocess)
        return (
            f"Processing result for {os.path.basename(abs_path)}: {status}",
            output_image_display,
        )

    except (ValueError, ConnectionError, ImportError, sqlite3.Error) as e:
        return (
            f"Failed to initialize ImageMetadataManager or connect to DB for tagging: {e}",
            None,
        )
    except (
        gr.Error
    ) as e:  # Catch errors from get_manager if it were used (not used here)
        return str(e), None
    except Exception as e:
        # Catch errors from Ollama, metadata writing etc. within process_single_image
        # These should ideally be caught within process_single_image and returned as status,
        # but this catches unexpected ones.
        return (
            f"An error occurred processing {os.path.basename(abs_path)}: {e}",
            output_image_display,  # Show image even on error if path is valid
        )
    finally:
        if manager:
            manager.close_db()


# --- View All Records Function Wrapper ---
def view_all_db_records(db_path):
    """Fetches all records from the database for display."""
    manager = None
    # Define columns explicitly for the empty DataFrame
    columns = ["ID", "Path", "Tags", "Status", "Timestamp", "Error", "Exists"]
    empty_df = pd.DataFrame(columns=columns)
    try:
        manager = get_manager(db_path)  # get_manager already checks if db exists

        # Query all relevant columns
        query = "SELECT id, original_path, ollama_tags, status, processed_timestamp, error_message FROM processed_images ORDER BY id"
        manager.cursor.execute(query)
        results = manager.cursor.fetchall()  # List of tuples

        if not results:
            return empty_df, "Database is empty or no records found."

        data_for_df = []
        total_records = len(results)

        for row in results:
            id_val, path, tags_json, status, timestamp, error_msg = row
            tags_list_from_db = []
            tags_str_display = ""
            if tags_json:
                try:
                    tags_list_from_db = json.loads(tags_json)
                    tags_str_display = ", ".join(tags_list_from_db)
                except json.JSONDecodeError:
                    tags_str_display = "(Invalid JSON in DB)"

            file_exists = os.path.exists(path)

            data_for_df.append(
                {
                    "ID": id_val,
                    "Path": path,
                    "Tags": tags_str_display,
                    "Status": status,
                    "Timestamp": timestamp,
                    "Error": error_msg
                    if error_msg
                    else "",  # Show empty string if no error
                    "Exists": file_exists,
                }
            )

        df = pd.DataFrame(data_for_df, columns=columns)  # Ensure column order
        status_msg = f"Loaded {total_records} record(s) from the database."
        return df, status_msg

    except gr.Error as e:  # Catch Gradio errors from get_manager
        return empty_df, str(e)
    except sqlite3.Error as e:
        return empty_df, f"Database query error: {e}"
    except Exception as e:
        return empty_df, f"An unexpected error occurred loading records: {e}"
    finally:
        if manager:
            manager.close_db()


# --- Helper Function for Gallery Selection ---
def handle_gallery_selection(evt: gr.SelectData | None = None):
    """
    Safely extracts the image path from the gallery selection event.
    Returns the path if an item is selected, otherwise returns an empty string.
    Handles cases where the event might be None or not the expected type.
    """
    # Check if evt is the expected type and has a value attribute
    if isinstance(evt, gr.SelectData) and hasattr(evt, "value") and evt.value:
        # evt.value is the path/URL of the selected image in the gallery
        return evt.value
    # Return empty string if no selection, evt is None, or evt is unexpected type
    return ""


# --- Gradio Interface Definition ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Image Tagger Database Interface")
    gr.Markdown(
        "Interact with the image metadata database generated by `img_tagger.py`."
    )

    # --- Database Path Input ---
    with gr.Row():
        db_path_input = gr.Textbox(
            label="Database Path",
            placeholder="Enter path and name of your db file",
            scale=3,
        )
        # Test connection button (optional, actions will test anyway)
        # test_db_button = gr.Button("Test Connection", scale=1)
        # test_db_output = gr.Textbox(label="Connection Status", interactive=False, scale=2)

    # Store db_path in state for other functions
    db_path_state = gr.State()
    # Update state when input changes
    db_path_input.change(lambda x: x, inputs=db_path_input, outputs=db_path_state)

    # --- Tabs for Different Actions ---
    with gr.Tabs():
        # --- Search Tab ---
        with gr.TabItem("Search Database"):
            gr.Markdown("Search for images based on tags stored in the database.")
            with gr.Row():
                search_tags_input = gr.Textbox(
                    label="Tags (comma-separated)",
                    placeholder="e.g., cat, outdoor, sunny",
                )
                search_location_input = gr.Textbox(
                    label="Location Keyword (optional)", placeholder="e.g., park, beach"
                )
            search_button = gr.Button("Search Images")
            search_status_output = gr.Textbox(label="Search Status", interactive=False)
            with gr.Row():
                search_results_gallery = gr.Gallery(
                    label="Found Images (Existing Files)",
                    show_label=True,
                    elem_id="search_gallery",
                    columns=6,
                    height=400,
                    object_fit="contain",
                    preview=True,  # Enable preview click
                )
            search_results_table = gr.DataFrame(
                label="Search Results Details", interactive=False, wrap=True
            )

            search_button.click(
                fn=search_db,
                inputs=[db_path_state, search_tags_input, search_location_input],
                outputs=[
                    search_results_gallery,
                    search_results_table,
                    search_status_output,
                ],
                api_name="search_images",  # Optional: for API access
            )

        # --- View All Records Tab --- ADDED
        with gr.TabItem("View All Records"):
            gr.Markdown("Load and view all records stored in the database file.")
            load_all_button = gr.Button("Load All Records")
            load_all_status_output = gr.Textbox(label="Load Status", interactive=False)
            all_records_table = gr.DataFrame(
                label="All Database Records",
                interactive=False,
                wrap=True,
                show_copy_button=True,
            )

            load_all_button.click(
                fn=view_all_db_records,
                inputs=[db_path_state],
                outputs=[all_records_table, load_all_status_output],
                api_name="view_all_records",
            )

        # --- View/Update/Delete Tab ---
        with gr.TabItem("Manage Records"):
            gr.Markdown("View, update (tags in DB only), or delete a specific record.")
            record_image_path_input = gr.Textbox(
                label="Image Path",
                placeholder="Enter the full path of the image record or select from gallery",
                interactive=True,  # Allow manual entry/paste
            )

            with gr.Row():
                view_button = gr.Button("View Record Details")
                delete_button = gr.Button("Delete DB Record", variant="stop")

            view_output_status = gr.Textbox(
                label="View/Delete Status", interactive=False
            )
            view_output_json = gr.JSON(
                label="Record Details"
            )  # Use JSON component for structured data

            gr.Markdown("---")
            gr.Markdown("### Update Tags (Database Only)")
            gr.Markdown(
                "Modify the `ollama_tags` stored in the database for the image path above. **This does not change the tags embedded in the image file itself.**"
            )
            update_tags_input = gr.Textbox(
                label="New Tags (comma-separated)",
                placeholder="e.g., feline, indoors, sleeping",
            )
            update_button = gr.Button("Update DB Tags")
            update_status_output = gr.Textbox(label="Update Status", interactive=False)

            # Link gallery selection to the image path input using the named handler
            search_results_gallery.select(
                fn=handle_gallery_selection,
                inputs=None,  # No extra inputs needed for the handler itself
                outputs=record_image_path_input,
                show_progress="hidden",  # Hide progress indicator for selection
            )

            # Button actions
            view_button.click(
                fn=view_db_record,
                inputs=[db_path_state, record_image_path_input],
                outputs=[view_output_json, view_output_status],
                api_name="view_record",
            )
            delete_button.click(
                fn=delete_db_record,
                inputs=[db_path_state, record_image_path_input],
                outputs=[view_output_status],  # Update status
                api_name="delete_record",
            )
            update_button.click(
                fn=update_db_tags,
                inputs=[db_path_state, record_image_path_input, update_tags_input],
                outputs=[update_status_output],
                api_name="update_tags",
            )

        # --- Tag Single Image Tab ---
        with gr.TabItem("Tag Single Image"):
            gr.Markdown(
                "Process a single image: Get tags from Ollama, embed tags into the image file (EXIF/PNG), and log/update the record in the database."
            )
            tag_single_image_path_input = gr.Textbox(
                label="Image Path",
                placeholder="Enter path to the image file or select from gallery",
                interactive=True,
            )
            tag_single_force_checkbox = gr.Checkbox(
                label="Force Reprocess",
                value=False,
                info="Check to re-tag and update record even if it already exists.",
            )
            tag_single_button = gr.Button("Process and Tag Image", variant="primary")
            tag_single_status_output = gr.Textbox(
                label="Processing Status", interactive=False
            )
            tag_single_image_output = gr.Image(
                label="Processed Image",
                type="filepath",
                interactive=False,  # Display only
            )

            # Link gallery selection to this input too, using the same handler
            search_results_gallery.select(
                fn=handle_gallery_selection,
                inputs=None,
                outputs=tag_single_image_path_input,
                show_progress="hidden",
            )

            tag_single_button.click(
                fn=tag_single_image_ui,
                inputs=[
                    db_path_state,
                    tag_single_image_path_input,
                    tag_single_force_checkbox,
                ],
                outputs=[tag_single_status_output, tag_single_image_output],
                api_name="tag_single_image",
            )


# --- Launch the Interface ---
if __name__ == "__main__":
    # Optional: Add code here to create a dummy DB for testing if needed
    # (Similar to the previous example, but using the ImageMetadataManager schema)
    # Make sure Ollama is running if you test the "Tag Single Image" feature.

    print("Launching Gradio Interface...")
    print("Make sure 'img_tagger.py' is accessible.")
    print("Ensure Ollama server is running for the 'Tag Single Image' feature.")

    # --- Cross-platform solution for allowed_paths ---
    # Get the user's home directory, works on Windows and Linux
    home_directory = os.path.expanduser("~")
    allowed_paths = [home_directory]
    print(
        f"Allowing Gradio access to user home directory and subfolders: {home_directory}"
    )
    # You could add more specific paths if needed, e.g.:
    # allowed_paths.append("/mnt/extra_drive/photos") # Example for Linux
    # allowed_paths.append(r"D:\Photos") # Example for Windows

    # Launch Gradio with the dynamically determined allowed path
    demo.launch(
        server_name="0.0.0.0",
        server_port=8504,
        allowed_paths=allowed_paths,  # Pass the list here
    )
