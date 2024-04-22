import cv2


def track_object():
    # Initialize video capture
    cap = cv2.VideoCapture(0)

    # Initialize tracker
    tracker = cv2.TrackerCSRT_create()

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to start camera.")
        return

    # Select ROI for tracker
    bbox = cv2.selectROI("Tracking", frame, False)
    tracker.init(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update tracker
        success, box = tracker.update(frame)

        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display result
        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

track_object()
