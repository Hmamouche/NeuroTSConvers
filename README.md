## Introduction
This repository represent a system of brain activity prediction using behavioral features.
The brain activity is measures through the BOLD signal, and the behavioral features represent verbal and non-verbal variables extracted during an experience of human-human and human-robot conversations conducted on several subjects.

The aim is to detect the behavioral features that are responsible for the activation of each brain area, by means of prediction.
A feature selection step is performed to select the input variables for the prediction of brain activity, then the most relevant input features are those how lead to the best prediction score.

## Data
The data are recorded during an experience of human-human and human-robot conversations
conducted on more than twenty participants, and divided into a set of
four sessions, where each one contains six conversations of sixty seconds each.
The conversations are performed under the same experimental conditions, but
alternatively with a human and a robot. The conversations are in a form of face
to face talk about images, in the same time, brain activity is obtained by fMRI,
1as well as eye movement of the participant, the audio files of both the agent
and the participants, and the videos of the agent.

## Brain activity data
The brain activity is obtained by fMRI, where the
BOLD signal is measured in different brain area during conversations. Each
conversation sppanes one minute, and the observations are spaced 1.205 seconds
apart. As a consequence, for each subject, the activities of 277 areas are recorded
by averaging voxels activity in each area.

## Analyzing audio files
*  First, and speech to text is performed then transcriptions are generated from audio files and the associated text.
*  We extract verbal and non-verbal features from the transcriptions, which are represented by the following features:
  Speech signal, IPU, Overlap, filled breaks, feedbacks, discourse items, particles_items, laughters, lexical richness.

## Analyzing video files
* We use OpenFace to extract the following features:
 * Facial Action Units
 * Landmarks
 * Head Pose Estimation
* And we use pre-trained deep learning models for emotions detection
* Finally, we construct time series using these features  by analyzing each image of the videos.

## Analyzing eyetracking data
* An eyetracking system is used to record the gaze coordinates. We process the data and compute the gradient,
then we project the coordinates on visual stimulation to record where the subject is looking in at each time step (face, eyes, mouth, else).
