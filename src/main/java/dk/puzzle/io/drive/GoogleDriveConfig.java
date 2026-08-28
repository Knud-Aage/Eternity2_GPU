package dk.puzzle.io.drive;

import com.google.api.client.auth.oauth2.Credential;
import com.google.api.client.extensions.java6.auth.oauth2.AuthorizationCodeInstalledApp;
import com.google.api.client.extensions.jetty.auth.oauth2.LocalServerReceiver;
import com.google.api.client.googleapis.auth.oauth2.GoogleAuthorizationCodeFlow;
import com.google.api.client.googleapis.auth.oauth2.GoogleClientSecrets;
import com.google.api.client.googleapis.javanet.GoogleNetHttpTransport;
import com.google.api.client.http.javanet.NetHttpTransport;
import com.google.api.client.json.JsonFactory;
import com.google.api.client.json.gson.GsonFactory;
import com.google.api.client.util.store.FileDataStoreFactory;
import com.google.api.services.drive.Drive;
import com.google.api.services.drive.DriveScopes;
import com.google.api.services.drive.model.FileList;
import com.google.api.services.drive.model.File;

import java.io.FileNotFoundException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Collections;
import java.util.List;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;


/**
 * Configuration utility for integrating Google Drive API services into the Eternity II solver.
 *
 * <p>This class manages the OAuth2 authentication flow, handles token persistence
 * in a local directory, and provides helper methods for interacting with Google Drive
 * storage, such as directory management for cloud-based checkpoints.</p>
 */
public class GoogleDriveConfig {
    private static final String APPLICATION_NAME = "Eternity II Solver";
    private static final JsonFactory JSON_FACTORY = GsonFactory.getDefaultInstance();

    private static final String TOKENS_DIRECTORY_PATH = "tokens";
    private static final List<String> SCOPES = Collections.singletonList(DriveScopes.DRIVE_FILE);
    private static final Logger logger = LogManager.getLogger(GoogleDriveConfig.class);


    /**
     * Initializes and returns an authorized Google Drive service instance.
     *
     * <p>The method orchestrates the following initialization steps:
     * <ol>
     *     <li>Loads client secrets from {@code /credentials.json} located in the resources directory.</li>
     *     <li>Sets up a {@link GoogleAuthorizationCodeFlow} with file-level access permissions.</li>
     *     <li>Triggers a local server receiver on port 8888 for user authorization.</li>
     *     <li>Persists or retrieves authorization tokens from the local {@code tokens} folder.</li>
     * </ol></p>
     *
     * @return An authorized {@link Drive} service instance ready for API operations.
     * @throws FileNotFoundException If the {@code credentials.json} file is missing from the classpath.
     * @throws Exception If an error occurs during the transport initialization or the OAuth2 authorization 
     *                   sequence.
     */
    public static Drive getDriveService() throws Exception {
        final NetHttpTransport HTTP_TRANSPORT = GoogleNetHttpTransport.newTrustedTransport();

        InputStream in = GoogleDriveConfig.class.getResourceAsStream("/credentials.json");
        if (in == null) {
            throw new FileNotFoundException(">>> [ERROR] Couldn't find /credentials.json. Does it lay in src/main/resources?");
        }

        GoogleClientSecrets clientSecrets = GoogleClientSecrets.load(JSON_FACTORY, new InputStreamReader(in));

        GoogleAuthorizationCodeFlow flow = new GoogleAuthorizationCodeFlow.Builder(
                HTTP_TRANSPORT, JSON_FACTORY, clientSecrets, SCOPES)
                .setDataStoreFactory(new FileDataStoreFactory(new java.io.File(TOKENS_DIRECTORY_PATH)))
                .setAccessType("offline")
                .build();

        LocalServerReceiver receiver = new LocalServerReceiver.Builder().setPort(8888).build();
        Credential credential = new AuthorizationCodeInstalledApp(flow, receiver).authorize("user");

        return new Drive.Builder(HTTP_TRANSPORT, JSON_FACTORY, credential)
                .setApplicationName(APPLICATION_NAME)
                .build();
    }

    /**
     * Searches for a folder by name in the user's Google Drive and creates it if it is not found.
     *
     * <p>This is used to ensure that puzzle records and logs are organized within a dedicated 
     * sub-folder in the cloud rather than cluttering the root directory.</p>
     *
     * @param driveService The authorized Google Drive service instance to use for the request.
     * @param folderName The display name of the folder to locate or initialize.
     * @return The unique Google Drive file ID for the folder.
     * @throws Exception If the API request fails, or if there is a conflict in retrieving or 
     *                   creating the folder metadata.
     */
    public static String getOrCreateFolder(Drive driveService, String folderName) throws Exception {
        String query = "mimeType='application/vnd.google-apps.folder' and name='" + folderName + "' and trashed=false";
        FileList result = driveService.files().list()
                .setQ(query)
                .setSpaces("drive")
                .setFields("files(id, name)")
                .execute();

        if (result.getFiles() != null && !result.getFiles().isEmpty()) {
            return result.getFiles().get(0).getId();
        }

        File folderMetadata = new File();
        folderMetadata.setName(folderName);
        folderMetadata.setMimeType("application/vnd.google-apps.folder"); // Dette gør filen til en mappe

        File folder = driveService.files().create(folderMetadata)
                .setFields("id")
                .execute();

        logger.info(">>> [GOOGLE DRIVE] Oprettede ny mappe i skyen: " + folderName);
        return folder.getId();
    }
}