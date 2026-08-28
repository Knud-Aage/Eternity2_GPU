package dk.puzzle.io.drive;

import com.google.api.client.http.FileContent;
import com.google.api.services.drive.Drive;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.File;
import java.nio.file.Files;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Collections;

/**
 * Mirrors every saved board to a Google Drive folder, as a one-line text record.
 *
 * <p><b>This is an optional convenience and is deliberately easy to remove.</b> It exists so a
 * long unattended run can be checked from a phone. Nothing in the solver depends on it, and it
 * can be taken out three ways, in increasing order of permanence:</p>
 *
 * <ol>
 *   <li><b>Turn it off:</b> set the environment variable {@code ETERNITY_DRIVE_UPLOAD=false}.
 *       No rebuild, no code change. This is also the default behaviour when the Drive
 *       credentials are simply absent -- see below.</li>
 *   <li><b>Comment it out:</b> delete the single {@code DriveUploader.uploadRecord(...)} call in
 *       {@code BlackwoodGpuRunner.evaluateAndMaybeSave} (and the matching one in
 *       {@code BlackwoodSolver} if that path is in use). Nothing else references this package.</li>
 *   <li><b>Delete it entirely:</b> remove this {@code dk.puzzle.io.drive} package, the call site
 *       above, and the three {@code com.google.*} dependencies from {@code pom.xml}. The project
 *       compiles and runs unchanged.</li>
 * </ol>
 *
 * <p>Credentials are never committed: {@code GoogleDriveConfig} reads {@code /credentials.json}
 * from the classpath and writes OAuth tokens to {@code tokens/}, both of which this repository's
 * {@code .gitignore} excludes. With no credentials present, the first upload attempt fails, is
 * logged once at WARN, and uploads disable themselves for the rest of the process -- so a fresh
 * clone runs correctly out of the box without any Drive setup at all.</p>
 */
public final class DriveUploader {

    private static final Logger logger = LogManager.getLogger(DriveUploader.class);

    /** Drive folder that records are written into; created on first use if absent. */
    private static final String DRIVE_FOLDER = "Blackwood";

    private static final boolean ENABLED =
            !"false".equalsIgnoreCase(System.getenv("ETERNITY_DRIVE_UPLOAD"));

    /**
     * Set once the first upload fails, so a run without credentials logs a single warning
     * instead of one per saved board (and doesn't repeatedly stall the search on a network
     * timeout it already knows will fail).
     */
    private static volatile boolean disabledAfterFailure = false;

    private DriveUploader() {
    }

    /**
     * Uploads a small text record (conflicts, source, timestamp, bucas link) for one saved board.
     *
     * <p>Best-effort by contract: this is called only after the local save has already succeeded,
     * and never throws. A Drive problem must not cost a result that is already safely on disk.</p>
     *
     * @param prefix        the saved board's filename prefix, e.g. {@code Errors12_Base250_024804_563}
     * @param conflicts     the completed board's edge-conflict count
     * @param completedLink bucas link to the completed board
     * @param source        which engine produced it, e.g. {@code "GPU"}
     */
    public static void uploadRecord(String prefix, int conflicts, String completedLink, String source) {
        if (!ENABLED || disabledAfterFailure) return;
        File linkFile = null;
        try {
            String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
            linkFile = File.createTempFile(prefix + "_", "_link.txt");
            Files.writeString(linkFile.toPath(), String.format(
                    "Edge Conflicts: %d%nSource: %s%nTime: %s%n%s%n",
                    conflicts, source, timestamp, completedLink));

            Drive driveService = GoogleDriveConfig.getDriveService();
            String folderId = GoogleDriveConfig.getOrCreateFolder(driveService, DRIVE_FOLDER);

            com.google.api.services.drive.model.File metadata = new com.google.api.services.drive.model.File();
            metadata.setName(linkFile.getName());
            metadata.setParents(Collections.singletonList(folderId));

            com.google.api.services.drive.model.File uploaded = driveService.files()
                    .create(metadata, new FileContent("text/plain", linkFile))
                    .setFields("id")
                    .execute();

            logger.info("Drive: uploaded {} ({} conflicts) to '{}', id {}",
                    prefix, conflicts, DRIVE_FOLDER, uploaded.getId());
        } catch (Exception e) {
            disabledAfterFailure = true;
            logger.warn("Drive upload failed for {} ({}); disabling further uploads for this run. "
                    + "This is harmless -- the board is already saved locally. Set ETERNITY_DRIVE_UPLOAD=false "
                    + "to skip this entirely, or see DriveUploader's javadoc to remove Drive support.",
                    prefix, e.getMessage());
        } finally {
            if (linkFile != null && linkFile.exists() && !linkFile.delete()) {
                linkFile.deleteOnExit();
            }
        }
    }
}
