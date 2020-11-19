from bert_embedding import BertEmbedding
import numpy as np
import re
import json

def extract_text_embeddings(input_dir, files, output_dir, fps, model="BERT"):
    embedding_model = create_embedding(model)

    for file in filenames:
        filename = os.path.join(input_dir, file + '.json')
        output_file = os.path.join(output_dir, file + '.npy')
        print('{}\t->\t{}'.format(file, output_file))

        text_features = encode_json_transcript(text_filename, embedding_model, fps)
        print("-- before upsampling:", text_features)
        # Upsample the text features from 10 FPS
        assert fps % 10 == 0
        target_len = len(text_features) * fps / 10
        cols = np.linspace(0, len(text_features), endpoint=False, num=target_len, dtype=int)
        text_features = text_features[cols]
        print("-- final shape:", text_features)

        np.save(output_file, text_features)

def create_embedding(name):
    if name == "BERT":
        return BertEmbedding(max_seq_length=100, model='bert_12_768_12', # COMMENT: will we ever change max_seq_length?
                             dataset_name='book_corpus_wiki_en_cased')
    # elif name == "FastText":
    #     return FastText()
    else:
        print(f"ERROR: Unknown embedding type '{name}'! Supported embeddings: 'BERT' and 'FastText'.")
        exit(-1)

def encode_json_transcript(text_filename, embedding_model)
    if isinstance(embedding_model, BertEmbedding):
        return _encode_json_transcript_with_bert(text_filename, embedding_model)
    # elif isinstance(embedding_model, FastText):
    #     return _encode_json_transcript_with_fasttext_10fps(text_filename, embedding_model)
    else:
        raise Exception("Unsupported embedding model with type" + type(embedding_model))


def _encode_json_transcript_with_bert(json_file, bert_model):
    """
    Parse json file and encode every word by BERT

    First, I am separating the text into sentences (because I believe that BERT works the best
     if applied for sentences) Then for each sentence, I collect timing information: which word
      lasted how long, then encode the whole sentence and finally I combine encodings and timings

    Example

    file = {"start_time": "0s", "end_time": "0.500s", "word": "I"},
           {"start_time": "0.5s", "end_time": "0.800s", "word": "love"},
           {"start_time": "0.800s", "end_time": "1s", "word": "you"}
    words = ["I", "love", "you"]
    timing = [1,1,1,1,1, 2,2,2,3,3]

    embed_words = [ [1,2,3] ,[2,3,4] ,[3,4,5] ]
    embed_final = [ [1,2,3] ,[1,2,3] ,[1,2,3] ,[1,2,3] ,[1,2,3],
                    [2,3,4] ,[2,3,4] ,[2,3,4] ,[3,4,5] ,[3,4,5] ]

    Args:
        json_file:        json of transcript of the speech signal by Google ASR
        bert_model:       BERT model

    Returns:
        feature_array:  an array of shape (n_frames, 773), where n_frames is the number of timeframes.
    
        NOTE: The transcription is processed at 10 FPS, so for a 60 second input 
              there will be 600 frames. 
        
        NOTE: The feature dimensionality is 773 because we add 5 extra features
              on top of the BERT dimensionality (768).
    """
    fillers = ["eh", "ah", "like", "kind of"]
    filler_encoding = bert_model(["eh, ah, like, kind of"])[0][1][0]
    delimiters = ['.', '!', '?']

    silence_encoding = np.array([-15 for i in range(768)]) # BERT has 768-dimensional features
    silence_extra_features = [0, 0, 0, 0, 0]
    
    elapsed_deciseconds = 0   
    feature_array = []
    
    # BERT requires the entire sentence instead of singular words

    # NOTE: The index 0 is reserved for silence, and the index 1 is reserved for filler words
    non_filler_words_in_sentence = [] 
    sentence_word_indices_list = [] # The index of the current word in the above vector for each frame
    sentence_extra_features_list = [] # The corresponding extra features
    
    with open(json_file, 'r') as file:
        transcription_segments = json.load(file)

    # The JSON files contain about a minute long segments
    for segment in transcription_segments: 
        segment_words = segment['alternatives'][0]['words']    

        for word_data in segment_words:                
            word = word_data['word']   

            # Word attributes: duration, speed, start time, end time 
            word_attributes = extract_word_attributes(word_data)

            # Get the index of the current word
            if word in fillers:
                # Fillers have word_idx 1
                curr_word_idx = 1 
            elif word[:-1] in fillers: # The last character of the word might be a delimiter
                curr_word_idx = 1

                # Here we explicitly check whether the delimiter signals the end of the sentence
                # For example, commas are not added to the word list
                if word[-1] in delimiters: 
                    non_filler_words_in_sentence.append(word[-1])
            else:
                # -> not a filler word

                # The first two indices are reserved for silence and fillers, 
                # therefore we start indexing from 2
                curr_word_idx = len(non_filler_words_in_sentence) + 2
                non_filler_words_in_sentence.append(word)

            # Process the silent frames before the word starts
            while elapsed_deciseconds < word_attributes['start_time']:
                elapsed_deciseconds += 1
                sentence_word_indices_list.append(0) # The idx 0 is reserved for silence
                sentence_extra_features_list.append(silence_extra_features)

            # Process the voiced frames           
            while elapsed_deciseconds < word_attributes['end_time']:
                elapsed_deciseconds += 1
                
                frame_features = extract_extra_features(
                                    word_attributes, elapsed_deciseconds)

                sentence_word_indices_list.append(curr_word_idx)
                sentence_extra_features_list.append(frame_features)

            # If the sentence is over, use bert to embed the words
            is_sentence_over = any([word[-1] == delimiter for delimiter in delimiters]) 

            if is_sentence_over:
                # Concatenate the words using space as a separator
                sentence = [' '.join(non_filler_words_in_sentence)]

                input_to_bert, encoded_words = bert_model(sentence)[0]

                if input_to_bert[-1] not in delimiters:
                    print("ERROR: missing delimiter in input to BERT!")
                    print("""\nNOTE: Please make sure that the last 'word'
                    field of each 'alternatives' segment in the input JSON file
                    ends with a punctuation mark (. ? or !)""")
                    print("The current sentence:", sentence)
                    print("The input to BERT:", bert_input)
                    exit(-1)

                # Add the silence/filler encodings at the reserved indices
                encoded_words = [silence_encoding] + [filler_encoding] + encoded_words

                # Frame-by-frame features of the entire sentence
                sentence_features = \
                    [ list(encoded_words[word_idx]) + sentence_extra_features_list[i]
                      for i, word_idx in enumerate(sentence_word_indices_list) ]

                # Add the sentence to the final feature list
                feature_array.extend(sentence_features)

                # Reset the sentence-level variables
                non_filler_words_in_sentence = []
                sentence_word_indices_list = []
                sentence_extra_features_list = []

    # In the GENEA dataset, some input transcriptions don't end with an
    # end-of-sentence token. We programmatically correct this error below.
    if not is_sentence_over:
        # The last sentence did not end with an end-of-sentence token
        # -> add one to the last word, then process the entire sentence
        non_filler_words_in_sentence[-1] += "."
        # Concatenate the words using space as a separator
        sentence = [' '.join(non_filler_words_in_sentence)]

        input_to_bert, encoded_words = bert_model(sentence)[0]

        if input_to_bert[-1] not in delimiters:
            print("ERROR: missing delimiter in input to BERT!")
            print("""\nNOTE: Please make sure that the last 'word'
            field of each 'alternatives' segment in the input JSON file
            ends with a punctuation mark (. ? or !)""")
            print("The current sentence:", sentence)
            print("The input to BERT:", bert_input)
            exit(-1)

        # Add the silence/filler encodings at the reserved indices
        encoded_words = [silence_encoding] + [filler_encoding] + encoded_words

        # Frame-by-frame features of the entire sentence
        sentence_features = \
            [ list(encoded_words[word_idx]) + sentence_extra_features_list[i]
                for i, word_idx in enumerate(sentence_word_indices_list) ]

        # Add the sentence to the final feature list
        feature_array.extend(sentence_features)


    if len(feature_array) != elapsed_deciseconds:
        print(f"ERROR: The number of frames in the encoded transcript ({len(feature_array)})") 
        print(f"       does not match the number of frames in the input ({elapsed_deciseconds})!")
        
        exit(-1)

    return np.array(feature_array)    


def json_time_to_deciseconds(time_in_text):
    """Convert timestamps from text representation to tenths of a second (e.g. '1.500s' to 15 deciseconds)."""
    # Remove the unit ('s' as in seconds) from the representation
    time_in_seconds = float(time_in_text.rstrip('s')) 

    return int(time_in_seconds * 10)

def extract_word_attributes(word_data):
    start_time = json_time_to_deciseconds(word_data['start_time'])
    end_time   = json_time_to_deciseconds(word_data['end_time'])
    duration   = end_time - start_time
    
    word = word_data['word']

    # Syllables per decisecond
    speed = count_syllables(word) / duration if duration > 0 else 10 # Because the text freq. is 10FPS

    attributes = { 'start_time': start_time, 'end_time': end_time,
                   'duration': duration, 'speed': speed }
    
    return attributes

def extract_extra_features(word_attributes, total_elapsed_time):
    """Return the word encoding and the additional features for the current frame as a list.
    The five additional features are: 
        
        1) elapsed time since the beginning of the current word 
        2) remaining time from the current word
        3) the duration of the current word
        4) the progress as the ratio 'elapsed_time / duration'
        5) the pronunciation speed of the current word (number of syllables per decisecond)

    Args:
        word_attributes:     A dictionary with word-level attributes. See extract_word_attributes() for details.
        total_elapsed_time:  The elapsed time since the beginning of the entire input sequence
    
    Returns: 
        frame_extra_features:  A list that contains the 5 additional features.
    """
    word_elapsed_time = total_elapsed_time - word_attributes['start_time']
    # The remaining time is calculated starting from the beginning of the frame - that's why we add 1
    word_remaining_time = word_attributes['duration'] - word_elapsed_time + 1 
    word_progress = word_elapsed_time / word_attributes['duration']        
  
    frame_extra_features = [ word_elapsed_time, 
                             word_remaining_time,
                             word_attributes['duration'], 
                             word_progress, 
                             word_attributes['speed'] ]

    return frame_extra_features

def count_syllables(word) :
    word = word.lower()

    # exception_add are words that need extra syllables
    # exception_del are words that need less syllables

    exception_add = ['serious','crucial']
    exception_del = ['fortunately','unfortunately']

    co_one = ['cool','coach','coat','coal','count','coin','coarse','coup','coif','cook','coign','coiffe','coof','court']
    co_two = ['coapt','coed','coinci']

    pre_one = ['preach']

    syls = 0 #added syllable number
    disc = 0 #discarded syllable number

    #1) if letters < 3 : return 1
    if len(word) <= 3 :
        syls = 1
        return syls

    #2) if doesn't end with "ted" or "tes" or "ses" or "ied" or "ies", discard "es" and "ed" at the end.
    # if it has only 1 vowel or 1 set of consecutive vowels, discard. (like "speed", "fled" etc.)

    if word[-2:] == "es" or word[-2:] == "ed" :
        doubleAndtripple_1 = len(re.findall(r'[eaoui][eaoui]',word))
        if doubleAndtripple_1 > 1 or len(re.findall(r'[eaoui][^eaoui]',word)) > 1 :
            if word[-3:] == "ted" or word[-3:] == "tes" or word[-3:] == "ses" or word[-3:] == "ied" or word[-3:] == "ies" :
                pass
            else :
                disc+=1

    #3) discard trailing "e", except where ending is "le"

    le_except = ['whole','mobile','pole','male','female','hale','pale','tale','sale','aisle','whale','while']

    if word[-1:] == "e" :
        if word[-2:] == "le" and word not in le_except :
            pass

        else :
            disc+=1

    #4) check if consecutive vowels exists, triplets or pairs, count them as one.

    doubleAndtripple = len(re.findall(r'[eaoui][eaoui]',word))
    tripple = len(re.findall(r'[eaoui][eaoui][eaoui]',word))
    disc+=doubleAndtripple + tripple

    #5) count remaining vowels in word.
    numVowels = len(re.findall(r'[eaoui]',word))

    #6) add one if starts with "mc"
    if word[:2] == "mc" :
        syls+=1

    #7) add one if ends with "y" but is not surrouned by vowel
    if word[-1:] == "y" and word[-2] not in "aeoui" :
        syls +=1

    #8) add one if "y" is surrounded by non-vowels and is not in the last word.

    for i,j in enumerate(word) :
        if j == "y" :
            if (i != 0) and (i != len(word)-1) :
                if word[i-1] not in "aeoui" and word[i+1] not in "aeoui" :
                    syls+=1

    #9) if starts with "tri-" or "bi-" and is followed by a vowel, add one.

    if word[:3] == "tri" and word[3] in "aeoui" :
        syls+=1

    if word[:2] == "bi" and word[2] in "aeoui" :
        syls+=1

    #10) if ends with "-ian", should be counted as two syllables, except for "-tian" and "-cian"

    if word[-3:] == "ian" :
    #and (word[-4:] != "cian" or word[-4:] != "tian") :
        if word[-4:] == "cian" or word[-4:] == "tian" :
            pass
        else :
            syls+=1

    #11) if starts with "co-" and is followed by a vowel, check if exists in the double syllable dictionary, if not, check if in single dictionary and act accordingly.

    if word[:2] == "co" and word[2] in 'eaoui' :

        if word[:4] in co_two or word[:5] in co_two or word[:6] in co_two :
            syls+=1
        elif word[:4] in co_one or word[:5] in co_one or word[:6] in co_one :
            pass
        else :
            syls+=1

    #12) if starts with "pre-" and is followed by a vowel, check if exists in the double syllable dictionary, if not, check if in single dictionary and act accordingly.

    if word[:3] == "pre" and word[3] in 'eaoui' :
        if word[:6] in pre_one :
            pass
        else :
            syls+=1

    #13) check for "-n't" and cross match with dictionary to add syllable.

    negative = ["doesn't", "isn't", "shouldn't", "couldn't","wouldn't"]

    if word[-3:] == "n't" :
        if word in negative :
            syls+=1
        else :
            pass

    #14) Handling the exceptional words.

    if word in exception_del :
        disc+=1

    if word in exception_add :
        syls+=1

    # calculate the output
    return numVowels - disc + syls

