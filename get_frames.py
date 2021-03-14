import cv2

scenes = ['Gura_Portitei_1', 'Gura_Portitei_2', 'Delta_Crisan_1']

def write_frames(folder, video, n):
    success = True
    count = 0

    while success:
        success, image = video.read()

        if image is not None and count % n == 0:
            image = image[4:1076,:,:]

            path = f'dataset/{folder}/images/frame_{count}.png'

            cv2.imwrite(path, image)

        count += 1


for scene in scenes:
    for dataset in ['train', 'test']:
        dataset_scene = f'{scene}_{dataset}'

        video_path  = f'videos/{dataset_scene}.mp4'

        video = cv2.VideoCapture(video_path)

        frames = write_frames(dataset_scene, video, 10)