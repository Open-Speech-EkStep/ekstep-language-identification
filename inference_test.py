import unittest
from inference import evaluation,create_output

model_path = './checkpoints/models-language_identification-Languages_vs_Songs-final_model.pt'


class TestEvaluation(unittest.TestCase):
    def test__file_with_no_sound_should_be_classified_as_vocal(self):
        self.assertEqual(evaluation('./Unit_testing_audio_files/Vocal_test_set/Malayalam/Malayalam_1_chunk_1.wav', model_path),
            dict({'confidence_score': {'vocal': '0.97222', 'songs': '0.02778'}}))

    def test__file_with_loud_music_only_should_be_classified_as_song(self):
        self.assertEqual(evaluation('./Unit_testing_audio_files/Songs_test_set/Punjabi/Punjabi_1_chunk_1.wav', model_path),
            dict({'confidence_score': {'vocal': '0.00013', 'songs': '0.99987'}}))

    def test__file_with_laughing_sound_should_be_classified_as_song(self):
        self.assertEqual(evaluation('./Unit_testing_audio_files/Songs_test_set/Punjabi/Punjabi_2_chunk_37.wav', model_path),
            dict({'confidence_score': {'vocal': '0.13867', 'songs': '0.86133'}}))

    def test__file_having_vocals_only_should_be_classified_as_vocal(self):
        self.assertEqual(evaluation('./Unit_testing_audio_files/Vocal_test_set/English/English_4_chunk_43.wav', model_path),
            dict({'confidence_score': {'vocal': '0.99998', 'songs': '0.00002'}}))

    def test__file_having_more_music_than_vocals_sound_should_be_classified_as_song(self):
        self.assertEqual(evaluation('./Unit_testing_audio_files/Songs_test_set/English/English_5_chunk_1.wav', model_path),
            dict({'confidence_score': {'vocal': '0.01493', 'songs': '0.98507'}}))

    def test__file_with_loud_music_and_vocals_should_be_classified_as_song(self):
        self.assertEqual(evaluation('./Unit_testing_audio_files/Songs_test_set/Punjabi/Punjabi_1_chunk_3.wav', model_path),
            dict({'confidence_score': {'vocal': '0.00000', 'songs': '1.00000'}}))

    def test__file_in_which_several_people_are_shouting_should_be_classified_as_song(self):
        self.assertEqual(evaluation('./Unit_testing_audio_files/Noisy_test_set/Debate/Debate_1_chunk_117.wav', model_path),
            dict({'confidence_score': {'vocal': '0.10760', 'songs': '0.89240'}}))

    def test__file_in_which_crowd_is_clapping_and_shouting_should_be_classified_as_song(self):
        self.assertEqual(evaluation('./Unit_testing_audio_files/Noisy_test_set/Noise/Noise_3_chunk_5.wav', model_path),
                         dict({'confidence_score': {'vocal': '0.00891', 'songs': '0.99109'}}))


class TestCreateOutput(unittest.TestCase):
    def test__input_confidence_scores_as_list_and_output_confidence_scores_as_dictionary(self):
        self.assertEqual(create_output(list(['0.00013', '0.99987'])), dict({'confidence_score': {'vocal': '0.00013', 'songs': '0.99987'}}))


if __name__ == '__main__':
    unittest.main()
