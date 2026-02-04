#!/usr/bin/env python3
"""
Google Drive Folder Downloader

This script downloads all files from a given Google Drive folder using the
Google Drive API. It handles authentication, pagination, and file downloads.

Requirements:
    pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib

Usage:
    python download_drive_folder.py "https://drive.google.com/drive/folders/YOUR_FOLDER_ID"
    python download_drive_folder.py "YOUR_FOLDER_ID" --output downloaded_videos
"""

import os
import io
import argparse
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Only request read access to Google Drive
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']


def authenticate():
    """
    Authenticate with Google Drive API and return the service object.
    Uses token.json for storing access/refresh tokens.
    """
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    # If no valid credentials, go through the OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists('credentials.json'):
                print("❌ Error: 'credentials.json' not found. Download it from Google Cloud Console.")
                return None
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save credentials for next time
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('drive', 'v3', credentials=creds)
    return service


def list_all_files(service, folder_id):
    """
    List all files in a Google Drive folder using pagination.

    Args:
        service: Authenticated Google Drive API service
        folder_id: ID of the folder to list files from

    Returns:
        List of file metadata dictionaries
    """
    files = []
    page_token = None
    print(f"🔍 Scanning folder ID: {folder_id}...")

    while True:
        response = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            spaces='drive',
            fields='nextPageToken, files(id, name, mimeType)',
            pageSize=100,
            pageToken=page_token
        ).execute()

        items = response.get('files', [])
        files.extend(items)
        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break

    return files


def download_file(service, file_id, file_name, output_folder):
    """
    Download a single file from Google Drive.

    Args:
        service: Authenticated Drive API service
        file_id: ID of the file to download
        file_name: Desired name for saved file
        output_folder: Local folder to save the file
    """
    file_path = os.path.join(output_folder, file_name)
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(file_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    print(f"⬇️  Downloading: {file_name}")
    try:
        while not done:
            status, done = downloader.next_chunk()
            # Optional: show progress for large files
            # print(f"Download {int(status.progress() * 100)}%")
    except Exception as e:
        print(f"❌ Error downloading {file_name}: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Download all files from a Google Drive folder")
    parser.add_argument("folder_url", help="Google Drive Folder URL or Folder ID")
    parser.add_argument("--output", default="downloads", help="Output directory for downloaded files")
    args = parser.parse_args()

    # Extract folder ID from URL if necessary
    if "folders/" in args.folder_url:
        folder_id = args.folder_url.split("folders/")[1].split("?")[0]
    else:
        folder_id = args.folder_url

    # Create output folder if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Authenticate with Google Drive
    service = authenticate()
    if not service:
        return

    # List all files in folder
    files = list_all_files(service, folder_id)
    print(f"✅ Found {len(files)} files in folder.")

    # Download each file
    for file in files:
        if file['mimeType'] == 'application/vnd.google-apps.folder':
            print(f"⏩ Skipping subfolder: {file['name']}")
            continue
        download_file(service, file['id'], file['name'], args.output)

    print("\n🎉 All downloads complete.")


if __name__ == "__main__":
    main()
