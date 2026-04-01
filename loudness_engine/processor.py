from pydub import AudioSegment

def apply_gain(file_path, gain_db, output_path):
    audio = AudioSegment.from_file(file_path)
    processed = audio.apply_gain(gain_db)
    processed.export(output_path, format="wav")