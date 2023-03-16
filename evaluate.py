from commu.midi_generator.generate_pipeline import MidiGenerationPipeline
from commu.preprocessor.encoder import encoder_utils
from commu.preprocessor.utils.constants import (
    BPM_INTERVAL,
    DEFAULT_POSITION_RESOLUTION,
    DEFAULT_TICKS_PER_BEAT,
    VELOCITY_INTERVAL,
    SIG_TIME_MAP,
    KEY_NUM_MAP
)
from commu.preprocessor.encoder.event_tokens import base_event, TOKEN_OFFSET
import numpy as np
import pandas as pd
import miditoolkit
import argparse
import math
from scipy import spatial
import re

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
NATURAL_MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
NUM_VELOCITY_BINS = int(128 / VELOCITY_INTERVAL)
DEFAULT_VELOCITY_BINS = np.linspace(2, 127, NUM_VELOCITY_BINS, dtype=np.int)
DEFAULT_POSITION_RESOLUTION = 128
event2word, word2event = encoder_utils.mk_remi_map()
TIME_SIG_MAP = {
    "4/4": 0,
    "3/4": 1,
    "6/8": 2,
    "12/8": 3,
}
pitch_ranges = {631: range(3, 39), 632: range(39, 51), 633: range(51, 63),
                634: range(63, 75), 635: range(75, 87), 636: range(87, 99),
                637: range(99, 131)}
CHORD_WORDS = base_event[TOKEN_OFFSET.CHORD_START.value - 2:]
scale_map = {"": (0, 4, 7),
             "7": (0, 4, 7, 10),
             "maj7": (0, 4, 7, 11),
             "sus4": (0, 5, 7),
             "7sus4": (0, 5, 7, 10),
             "+": (0, 4, 8),
             "dim": (0, 3, 6),
             "dim7": (0, 3, 6, 9),
             "m": (0, 3, 7),
             "m7": (0, 3, 7, 10),
             "m7b5": (0, 3, 6, 10),
             "sus2": (0, 2, 7),
             "add2": (0, 2, 4, 7),
             "madd2": (0, 2, 3, 7),
             "m6": (0, 3, 7, 9),
             "6": (0, 4, 7, 9)}

chord_lst = [
    "a",
    "a#",
    "b",
    "c",
    "c#",
    "d",
    "d#",
    "e",
    "f",
    "f#",
    "g",
    "g#",
]


class Event(object):
    def __init__(self, name, time, value, text):
        self.name = name
        self.time = time
        self.value = value
        self.text = text

    def __repr__(self):
        return "Event(name={}, time={}, value={}, text={})".format(
            self.name, self.time, self.value, self.text
        )


def word_to_event(words, word2event):
    events = []
    for word in words:
        try:
            event_name, event_value = word2event[word].split("_")
        except KeyError:
            if word == 1:
                # 따로 디코딩 되지 않는 EOS
                continue
            else:
                print(f"OOV: {word}")
            continue
        events.append(Event(event_name, None, event_value, None))
    return events


def isPitchCorrect(pitch, meta):
    if not 3 <= pitch <= 130:
        print("Error encountered {}".format(pitch))
    return pitch in pitch_ranges[meta[3]]


def isVelocityCorrect(velocity, encoded_meta):
    return encoded_meta[7] - 523 <= velocity <= encoded_meta[8] - 523


def mk_chord_scale_map(chord_words = CHORD_WORDS):
    chord2symbol = {k: v for k, v in zip(chord_lst, range(12))}
    chord2scale = dict()
    for event in chord_words:
        if "#" in event:
            root = event.split('_')[1][0] + re.findall(r"([#])", event)[0]
            quality = event.split('_')[1].replace(root, "")
        else:
            root = event.split('_')[1][0]
            quality = event.split('_')[1][1:]
        if root != "N":
            diff = chord2symbol[root] - chord2symbol['c']
            scale = [(k + diff) % 12 for k in scale_map[quality]]
        else:
            scale = []
        chord2scale["Chord_" + root + quality] = scale
    return chord2scale


def in_harmoney_rate(notes, chord_progression, key_num):
    scale = key_num // 12
    diff = key_num % 12
    key_scale = []
    if scale == 0:
        for i in MAJOR_SCALE:
            key_scale.append((i + diff) % 12)
    else:
        for i in NATURAL_MINOR_SCALE:
            key_scale.append((i + diff) % 12)

    p, n = 0, 0
    pn = [0] * len(notes)
    chord_times = [240 * i for i in range(len(chord_progression) + 1)]  # 240 ticks per 8th note
    chord_scale_map = mk_chord_scale_map()
    for note_idx, note in enumerate(notes):
        for idx, time in enumerate(chord_times[:-1]):
            next_time = chord_times[idx + 1]
            if (time < note.start < next_time) or (  # | ----note----
                    time < note.end < next_time) or (  # ----note---- |
                    note.start < time and note.end > next_time):  # --|--note--|--
                if note.pitch % 12 in key_scale:
                    pn[note_idx] += 0
                elif note.pitch % 12 in chord_scale_map['Chord_' + chord_progression[idx].lower()]:
                    pn[note_idx] += 0
                else:
                    pn[note_idx] += 1
            else:
                pass
    for i in pn:
        if i:
            n += 1
        else:
            p += 1
    pn = p / (p + n)
    return p, n, pn


def extract_attr_vector_per_bar(note, attr: str = None):
    # set parameters by attribute
    dim, index_start, index_end = None, None, None
    if attr == 'chroma': # <NOTE PITCH>
        dim, index_start, index_end = 12, 3, 130
    elif attr == 'grv': # <DURATION>
        dim, index_start, index_end = 128, 304, 431

    total_out = []
    for n in note:
        if n == 2:
            out = [0] * dim
            total_out.append(out)
        if index_start <= n <= index_end:
            n_shift = n - index_start # shift to start from 0
            ind = n_shift % 12 if attr == "chroma" else n_shift
            out[int(ind)] += 1

    return total_out


def get_dist(note_1, note_2, attr: str = None):
    # compute attr vectors
    vec_1 = extract_attr_vector_per_bar(note_1, attr=attr)
    vec_2 = extract_attr_vector_per_bar(note_2, attr=attr)
    assert len(vec_1) == len(vec_2), "Inputs must have the same shape!"
    # compute cosine similarity between attr. vectors
    out = []
    for v1, v2 in zip(vec_1, vec_2):
        if sum(v1) * sum(v2) == 0:
            continue
        dist = spatial.distance.cosine(v1, v2)
        out.append(dist)

    return np.array(out).mean() # mean over bar


def calChromaSim(a_notes, b_notes):
    ret = get_dist(a_notes, b_notes, attr='chroma')
    return ret


def calGroovingSim(a_notes, b_notes):
    ret = get_dist(a_notes, b_notes, attr='grv')
    return ret


def main(checkpoint_path, meta_data_path, eval_diversity=False):
    pipeline = MidiGenerationPipeline()

    # Model Initialize
    pipeline.initialize_model(model_arguments={'checkpoint_dir': checkpoint_path})
    pipeline.initialize_generation()

    inference_cfg = pipeline.model_initialize_task.inference_cfg
    model = pipeline.model_initialize_task.execute()

    # Validation Meta Data load
    metas = pd.read_csv(meta_data_path)
    del metas['Unnamed: 0']

    CP, CV, PH, NH, Diversity = 0, 0, 0, 0, 0
    num_notes = 0
    for idx, meta in enumerate(metas.iloc):
        input_ = dict(meta)
        input_['rhythm'] = 'standard'
        input_['output_dir'] = './out1'
        input_['num_generate'] = 10 if eval_diversity else 1
        input_['top_k'] = 32
        input_['temperature'] = 0.95
        input_['chord_progression'] = str('-'.join(eval(input_['chord_progressions'])[0]))
        input_.pop('chord_progressions')
        
        pipeline.preprocess_task.input_data = None
        encoded_meta = pipeline.preprocess_task.execute(input_)
        input_data = pipeline.preprocess_task.input_data

        pipeline.inference_task(
            model=model,
            input_data=input_data,
            inference_cfg=inference_cfg
        )

        sequences = pipeline.inference_task.execute(encoded_meta)

        ### 평가 ###
        same_meta_output = []
        for idx2, seq in enumerate(sequences):
            encoded_meta = seq[1:12]
            event_sequences = seq[13:]
            events = word_to_event(event_sequences, word2event)

            temp_notes = []
            temp_chords = []

            time_sig = meta['time_signature']
            numerator = int(time_sig.split("/")[0])
            denominator = int(time_sig.split("/")[1])
            beats_per_bar = int(numerator / denominator * 4)

            ticks_per_bar = DEFAULT_TICKS_PER_BEAT * beats_per_bar

            duration_bins = np.arange(
                int(ticks_per_bar / DEFAULT_POSITION_RESOLUTION),
                ticks_per_bar + 1,
                int(ticks_per_bar / DEFAULT_POSITION_RESOLUTION),
                dtype=int,
            )
            for i in range(len(events) - 3):
                if events[i].name == "Bar" and i > 0:
                    temp_notes.append("Bar")
                    temp_chords.append("Bar")
                elif (
                    events[i].name == "Position"
                    and events[i + 1].name == "Note Velocity"
                    and events[i + 2].name == "Note On"
                    and events[i + 3].name == "Note Duration"
                ):
                    position = int(events[i].value.split("/")[0]) - 1

                    index = int(events[i + 1].value)
                    velocity = int(events[i + 1].value) + 131

                    pitch = int(events[i + 2].value) + 3

                    index = int(events[i + 3].value)
                    duration = duration_bins[index]

                    temp_notes.append([position, velocity, pitch, duration])
                elif events[i].name == "Position" and events[i + 1].name == "Chord":
                    position = int(events[i].value.split("/")[0]) - 1
                    temp_chords.append([position, events[i + 1].value])

            notes = []
            current_bar = 0
            for note in temp_notes:
                if note == "Bar":
                    current_bar += 1
                else:
                    position, velocity, pitch, duration = note

                    current_bar_st = current_bar * ticks_per_bar
                    current_bar_et = (current_bar + 1) * ticks_per_bar

                    flags = np.linspace(
                        int(current_bar_st),
                        int(current_bar_et),
                        int(DEFAULT_POSITION_RESOLUTION),
                        endpoint=False,
                        dtype=int,
                    )

                    st = flags[position]
                    et = st + duration
                    notes.append(miditoolkit.Note(velocity, pitch, st, et))

            # 코드 체크
            if len(temp_chords) > 0:
                chords = []
                current_bar = 0
                for chord in temp_chords:
                    if chord == "Bar":
                        current_bar += 1
                    else:
                        position, value = chord
                        current_bar_st = current_bar * ticks_per_bar
                        current_bar_et = (current_bar + 1) * ticks_per_bar
                        flags = np.linspace(
                            current_bar_st, current_bar_et, DEFAULT_POSITION_RESOLUTION, endpoint=False, dtype=int
                        )
                        st = flags[position]
                        chords.append([st, value])
                
                # CP, CV for each note sequences
                pitchsum = 0
                for note in notes:
                    pitchsum += note.pitch
                    if isVelocityCorrect(note.velocity, encoded_meta):
                        CV += 1
                
                avgpitch = int(pitchsum / len(notes))
                if isPitchCorrect(avgpitch, encoded_meta):
                    CP += 1
                num_notes += len(notes)

            TP, TN, _ = in_harmoney_rate(notes, chord_progression=eval(dict(meta)['chord_progressions'])[0],
                                         key_num=encoded_meta[1]-601)
            PH += TP
            NH += TN
            same_meta_output.append(notes)

        # Diversity
        if eval_diversity:
            tmp_Diversity = 0
            num_nan = 0
            for i in range(len(sequences)):
                for j in range(i+1, len(sequences)):
                    if np.isnan(math.sqrt(((1-calChromaSim(sequences[i], sequences[j]))**2 + (1-calGroovingSim(sequences[i], sequences[j]))**2) / 2)):
                        num_nan += 1
                        continue
                    tmp_Diversity += math.sqrt(((1-calChromaSim(sequences[i], sequences[j]))**2
                                           + (1-calGroovingSim(sequences[i], sequences[j]))**2)
                                           / 2)
            tmp_Diversity /= len(sequences) * (len(sequences) - 1) / 2 - num_nan
            Diversity += tmp_Diversity

        print(idx, '/', len(metas), "\tCP", CP / 10 / (idx + 1), "\tCV", CV / num_notes, "\tCH", PH / (PH+NH), flush = True)
        if eval_diversity:
            print("DIV", Diversity / (idx + 1), flush = True)

    Diversity /= len(metas)
    CP /= 10  * len(metas)
    CV /= num_notes
    CH = PH / (PH + NH)

    print("CP:{0:.4}, CV:{1:.4}, CH:{2:.4}".format(CP, CV, CH))
    if eval_diversity:
        print("Diversity:{0:.4}".format(Diversity))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--checkpoint_dir", type=str, required=True)
    arg_parser.add_argument("--val_meta_dir", type=str, required=True)
    arg_parser.add_argument("--eval_diversity", type=bool, required=False)

    args = arg_parser.parse_args()

    main(args.checkpoint_dir, args.val_meta_dir, args.eval_diversity)