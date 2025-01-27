from TTS.api import TTS
import odak
import sys


def main():
    files = sorted(odak.tools.list_files('./', key = '*.md'))
    files = ['computer_generated_holography.md']
    tts = TTS(model_name = "tts_models/en/jenny/jenny", progress_bar = True, gpu = True)
    cache_fn = 'cache.txt'
    wav_file = 'cache.wav'
    for file in files:
        print(file)
        cmd = ['cp', str(file), cache_fn]
        odak.tools.shell_command(cmd)
        f = open(cache_fn, 'r+')
        contents_list = f.readlines()
        f.close()
        if contents_list == []:
            sys.exit()
        mp3_file = str(file.replace('.md', '.mp3'))
        contents = clear_text(contents_list)
        tts.tts_to_file(
                        text = contents,
                        file_path = wav_file,
                        emotion = 'Happy',
                        speed = 0.8
                       )
        cmd = ['ffmpeg', '-i', wav_file, mp3_file, '-y']
        odak.tools.shell_command(cmd)
        cmd = ['mv', mp3_file, './media/']
        odak.tools.shell_command(cmd) 
        cmd = ['rm', cache_fn]
        odak.tools.shell_command(cmd)
        cmd = ['rm', wav_file]
        odak.tools.shell_command(cmd)
            

def clear_text(text):
    found_ids = []
    for item_id, item in enumerate(text):
        for word in ['<script>', '<div>', '</div>', '</script>', 'mesh3d', 'showlegend']:
            if item.find(word) != -1:
                found_ids.append(item_id)
    new_list = []
    for item_id, item in enumerate(text):
        if not item_id in found_ids:
            new_list.append(item)
    text = new_list
    output_text = ''.join(text)
    output_text = output_text.replace('???', '')
    output_text = output_text.replace('Narrate section', '')
    output_text = output_text.replace(':material-alert-decagram:{ .mdx-pulse title="Too important!" }', 'Too important!')
    output_text = output_text.replace(':octicons-beaker-24:', '')
    output_text = output_text.replace(':octicons-info-24:', '')
    output_text = output_text.replace('quote end', '')
    output_text = output_text.replace('question end', '')
    output_text = output_text.replace('information end', '')
    output_text = output_text.replace('success end', '')
    output_text = output_text.replace('Warning end', '')
    output_text = output_text.replace('!!!', '')
    output_text = output_text.replace('#', '')
    output_text = output_text.replace('##', '')
    output_text = output_text.replace('###', '')
    output_text = output_text.replace('####', '')
    output_text = output_text.replace('$', '')
    output_text = output_text.replace('$$', '')
    output_text = output_text.replace('*', '')
    output_text = output_text.replace('**', '')
    return output_text


if __name__ == '__main__':
    sys.exit(main())
