from time import time
from pymongo import MongoClient
from scipy.sparse import dok_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np

MONGO_URL = 'mongodb://um.media.mit.edu:27017/super-glue'
db = MongoClient(MONGO_URL).get_default_database()

nlp_data = db['nlp_data'].find_one()
stopwords = nlp_data["stopwords"]
vocab = nlp_data["vocab_non_stemmed"]
vocab_dict = nlp_data["vocab_non_stemmed_dict"]

DAY = 86400000
HOUR = 3600000

def millis():
    return int(round(time() * 1000))
def millis_since(days=1):
    return millis() - days*DAY
def millis_since_hours(hours):
    return millis()-HOUR*hours

def get_vectors(media):
    segs = media["story_segments"]
    media_url = "media_url_no_comm"
    if not media["module_reports"]["commercial_skip_module"]["removed_commercials"]:
        media_url = "media_url"
    segs_vectors = []
    file_name = lambda x:''.join(x.split('.')[4:])
    for i in range (len(segs)):
        start = segs[i]["start"]
        end = segs[i]["end"]
        thumb = "/static/images/blank.jpg"
        if "thumbnail_image" in segs[i]:
            thumb = segs[i]["thumbnail_image"]
        url = "%s#t=%.2f,%.2f"%(media[media_url],start/1000.0,end/1000.0)
        air_date = media["date_added"]
        length = float(end)-float(start)
        if "word_count" in segs[i]:
            vector = segs[i]["word_count"]
            if len(vector.keys())>3 and length>4000:
                segs_vectors.append({
                    "start":start,
                    "end":end,
                    "url":url,
                    "channel":media["channel"],
                    "length":length,
                    "date":air_date,
                    "thumbnail":thumb,
                    "media_id": str(media["_id"]),
                    "segment_index": i,
                    "vector":vector})
    return segs_vectors

def get_all_segments(timeframe = '1'):
    all_media_has_segments = db['media'].find(
        {"date_added": {"$gt": millis_since(timeframe)},
         "story_segments":{"$exists": True},"is_news":{"$eq": True}})
    num_of_videos = all_media_has_segments.count()
    print "%d videos"%num_of_videos
    all_segments = []
    for media in all_media_has_segments:
        segs = get_vectors(media)
        all_segments.extend(segs)
    print "%d total segments"%len(all_segments)
    return all_segments

def get_tf_matrix(all_segments):
    vectors = dok_matrix((len(all_segments), len(vocab)), dtype=np.int32)
    for i, seg in enumerate(all_segments):
        vec = seg["vector"]
        for ind in vec.keys():
            vectors[i, int(ind)] = int(vec[ind])
        seg.pop("vector")
    return vectors

def get_data(timeframe=1):
    all_segments = get_all_segments(timeframe)
    tf = get_tf_matrix(all_segments)
    transformer = TfidfTransformer(norm=u'l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
    tf_idf = transformer.fit_transform(tf)
    words_sizes_tf = tf.sum(axis=0)
    words_sizes_tfidf = tf_idf.sum(axis=0)
    return {
    'all_segments': all_segments,
    'tf':tf,
    'tf_idf':tf_idf,
    'words_sizes_tf': words_sizes_tf,
    'words_sizes_tfidf': words_sizes_tfidf,
    'stopwords': stopwords,
    'vocab': vocab,
    'vocab_dict': vocab_dict,
    'timeframe':timeframe
    }
