import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import engine as e
from html import escape

st.set_page_config(page_title='CineMood',page_icon='🎬',layout='wide')
A=Path(__file__).parent/'assets'
@st.cache_resource
def load():
    df=pd.read_json(A/'cinemood_movies.json')
    df['release_date']=pd.to_datetime(df.release_date,errors='coerce')
    # Refit from exported features for portability across library versions.
    v=TfidfVectorizer(stop_words='english',ngram_range=(1,2),min_df=2,max_df=.90,max_features=20000,sublinear_tf=True)
    m=v.fit_transform(df.combined_features.fillna(''))
    return df,v,m,cosine_similarity(m)
e.df,e.tfidf,e.tfidf_matrix,e.similarity_matrix=load()
df=e.df
import base64
hero_path = A / 'Auroras sobre montañas violetas.png'
hero_image = base64.b64encode(hero_path.read_bytes()).decode('ascii') if hero_path.is_file() else ''
st.markdown("""
<style>
.block-container {padding-top:2.5rem;max-width:1250px;}
.cine-hero {overflow:hidden;border-radius:22px;background:#191426;border:1px solid #352746;margin-bottom:1.5rem;}
.cine-landscape {display:block;width:100%;aspect-ratio: 21/9;object-fit:cover;}
.cine-copy {padding:1.5rem 2rem 1.8rem;}
.cine-kicker {color:#d8b4fe;font-size:.8rem;font-weight:700;letter-spacing:.06em;}
.cine-hero h1 {color:#fff;font-size:clamp(2rem,4vw,3rem);margin:.4rem 0;}
.cine-tagline {color:#f0e8ff;font-size:1.3rem;font-weight:600;margin:.5rem 0;}
.cine-description {color:#d1c4e9;margin:.7rem 0;}
.cine-stats {display:flex;flex-wrap:wrap;gap:2rem;margin-top:1rem;}
.cine-stats strong {display:block;color:#c084fc;font-size:1.65rem;}
.cine-stats span {color:#e2d9ed;font-size:.9rem;}
[data-testid="stMetric"] {background:#171b2b;border:1px solid #343047;padding:1rem;border-radius:14px;}
[data-testid="stImage"] img {border-radius:12px;}
@media(max-width:600px){.cine-copy{padding:1rem;}.cine-stats{gap:1rem;}}
.cine-poster {display:block;width:100%;aspect-ratio:2/3;object-fit:cover;border-radius:12px;background:#211932;transition:filter .18s ease;}
.cine-poster:hover {filter:brightness(1.12);}
.cine-placeholder {display:grid;place-content:center;text-align:center;color:#c4b5fd;}
@media(prefers-reduced-motion:reduce){.cine-poster{transition:none;}}
</style>
""", unsafe_allow_html=True)
hero_banner = f'<img class="cine-landscape" src="data:image/png;base64,{hero_image}" alt="Purple and blue auroras over mountains">' if hero_image else ''
st.markdown(f"""<div class="cine-hero">{hero_banner}<div class="cine-copy">
<div class="cine-kicker">YOUR PERSONAL MOVIE DISCOVERY ENGINE</div>
<h1>🎬 CineMood</h1>
<div class="cine-tagline">Your next movie starts with a feeling.</div>
<div class="cine-description">Discover stories that match your mood, personality and taste.</div>
<div class="cine-stats"><div><strong>{len(df):,}</strong><span>Movies</span></div><div><strong>AI</strong><span>Recommendations</span></div><div><strong>10</strong><span>Discovery tools</span></div></div>
</div></div>""", unsafe_allow_html=True)
st.session_state.setdefault('watchlist',[])
# Native Streamlit navigation: preserves session state and requires no extra packages.
st.session_state.setdefault('cine_page', 'Discover')
nav_groups = [
    ('DISCOVER', [('search', 'Discover'), ('sentiment_satisfied', 'Mood Match'), ('movie', 'Similar Movies'), ('casino', 'Surprise Me'), ('diamond', 'Hidden Gems'), ('emoji_events', 'Top Rated')]),
    ('YOUR CINEMA', [('favorite', 'For You'), ('bookmark', 'My Watchlist')]),
    ('EXPLORE & ANALYZE', [('compare_arrows', 'Compare'), ('bar_chart', 'Analytics')]),
]
def navigate(destination):
    st.session_state.cine_page = destination

st.markdown("""
<style>
[data-testid="stSidebar"] {background:#171827;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:.22rem;}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {padding-bottom:1rem;}
[data-testid="stSidebar"] .stButton button {
    width:100%; justify-content:flex-start !important; text-align:left !important;
    border:1px solid transparent; border-radius:9px; background:transparent;
    color:#e2e2eb; padding:.42rem .65rem; min-height:2.35rem;
    transition:background-color .15s ease,color .15s ease,border-color .15s ease;
}
[data-testid="stSidebar"] .stButton button > div {justify-content:flex-start !important;text-align:left !important;}
[data-testid="stSidebar"] .stButton button p {text-align:left !important;margin:0;font-size:.9rem;}
[data-testid="stSidebar"] .stButton button [data-testid="stIconMaterial"] {color:#aaa9bd;font-size:1.15rem;flex-shrink:0;}
[data-testid="stSidebar"] .stButton button:hover {background:#30243f;color:#e9d5ff;border-color:#49345f;}
[data-testid="stSidebar"] .stButton button[kind="primary"],
[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"] {
    background:#482866;color:#f3e8ff;border-color:#76509c;box-shadow:inset 3px 0 #c084fc;
}
[data-testid="stSidebar"] .stButton button[kind="primary"] [data-testid="stIconMaterial"] {color:#d8b4fe;}
[data-testid="stSidebar"] .stButton button:focus-visible {outline:2px solid #c084fc;outline-offset:1px;}
.cine-nav-brand {font-size:1.55rem;font-weight:750;color:#f4edff;line-height:1.4;}
.cine-nav-sub {font-size:.8rem;color:#a5a0b5;line-height:1.5;}
.cine-nav-group {display:block;box-sizing:border-box;padding:.85rem .65rem .35rem;line-height:1.4;font-size:.68rem;font-weight:700;letter-spacing:.09em;color:#a5a0b5;}
.cine-nav-footer {border-top:1px solid #39334b;padding-top:.8rem;color:#a5a0b5;font-size:.75rem;line-height:1.7;}
@media(prefers-reduced-motion:reduce){[data-testid="stSidebar"] .stButton button{transition:none;}}
</style>
""", unsafe_allow_html=True)
with st.sidebar:
    st.markdown('<div class="cine-nav-brand">CineMood</div><div class="cine-nav-sub">Your personal cinema</div>', unsafe_allow_html=True)
    for group, entries in nav_groups:
        st.markdown(f'<div class="cine-nav-group">{group}</div>', unsafe_allow_html=True)
        for symbol, destination in entries:
            selected = st.session_state.cine_page == destination
            st.button(destination, icon=f':material/{symbol}:',
                      key=f'cine-nav-{destination}', type='primary' if selected else 'secondary',
                      use_container_width=True, on_click=navigate, args=(destination,))
    st.markdown(f'<div class="cine-nav-footer">{len(df):,} movies · Powered by TMDB<br>{len(st.session_state.watchlist)} saved movies</div>', unsafe_allow_html=True)
page = st.session_state.cine_page
page_intro = {
    'Discover': ('Discover your next favorite', 'Describe what you want to watch or refine your preferences.'),
    'Mood Match': ('How do you want to feel tonight?', 'Every feeling has a story.'),
    'Similar Movies': ('More movies like your favorites', 'Select a reference movie to explore related stories.'),
    'Surprise Me': ('🎲 Surprise Me', 'Sometimes the best stories are the ones you never expected.'),
    'Hidden Gems': ('💎 Hidden Gems', 'Extraordinary stories beyond the spotlight.'),
    'Top Rated': ('🏆 Top Rated', 'Discover the highest-rated stories in our collection.'),
    'For You': ('💜 For You', 'Every movie you like brings us closer to your next favorite.'),
    'My Watchlist': ('Your personal collection', 'Keep your discoveries together during this session.'),
    'Compare': ('Compare your picks', 'Explore up to four movies side by side.'),
    'Analytics': ('Inside the recommendation engine', 'Explore the dataset, model benchmark and quality checks.')
}
st.header(page_intro[page][0])
st.caption(page_intro[page][1])
labels={f'{r.title} ({r.year}) · {r.id}':i for i,r in df.iterrows()}

def options(col):
    return sorted({x for values in df[col] for x in values if x})

def toggle_watchlist(movie_id):
    if movie_id in st.session_state.watchlist:
        st.session_state.watchlist.remove(movie_id)
    else:
        st.session_state.watchlist.append(movie_id)


from urllib.request import urlopen, Request
from urllib.parse import urlencode
from urllib.error import URLError
import json
import re

@st.cache_data(ttl=86400, show_spinner=False)
def find_trailer(movie_id, _api_key):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}/videos?' + urlencode({'api_key': _api_key, 'language': 'en-US'})
    with urlopen(Request(url, headers={'Accept': 'application/json'}), timeout=10) as response:
        data = json.load(response)
    videos = [v for v in data.get('results', []) if v.get('site') == 'YouTube' and v.get('type') == 'Trailer' and re.fullmatch(r'[A-Za-z0-9_-]{11}', str(v.get('key', '')))]
    videos.sort(key=lambda v: (bool(v.get('official')), v.get('published_at') or ''), reverse=True)
    return videos[0]['key'] if videos else None


def show_trailer(r):
    movie_id = int(r.id)
    fallback = 'https://www.youtube.com/results?' + urlencode({'search_query': f'{r.title} {r.year} official trailer'})
    if st.toggle('▶ Watch Trailer', key=f'trailer-{movie_id}'):
        try:
            api_key = st.secrets.get('TMDB_API_KEY', '')
            if not api_key:
                st.info('Trailer service is not configured.')
            else:
                with st.spinner('Finding the trailer...'):
                    video_key = find_trailer(movie_id, api_key)
                if video_key:
                    video_url = f'https://www.youtube.com/watch?v={video_key}'
                    st.video(video_url)
                    st.link_button('Open trailer on YouTube', video_url)
                else:
                    st.info('No English-language YouTube trailer found in TMDB.')
        except (URLError, TimeoutError, ValueError, OSError, KeyError):
            st.warning('Trailer service is temporarily unavailable.')
        st.link_button('Search official trailer on YouTube', fallback)


@st.dialog("Movie details", width="large")
def movie_details(r, mood=None, reference=None, query=''):
    a,b=st.columns([1,2])
    with a:
        if isinstance(r.poster_url,str) and r.poster_url.startswith('https://'):
            st.image(r.poster_url, use_container_width=True)
    with b:
        st.header(f'{r.title} ({r.year})')
        st.caption(f'★ {r.vote_average:.1f}/10 · {r.runtime:g} min · {r.vote_count:,} votes')
        meta=[str(x) for x in (r.certification,r.original_language) if pd.notna(x) and str(x).strip().lower() not in ('none','nan','')]
        if meta: st.caption(' · '.join(meta))
        st.write(' · '.join(r.genres))
        if 'match_score' in r and pd.notna(r.match_score):
            st.markdown(f':green[**{r.match_score:.1f}/100 Match index**]')
        st.caption(f'Dominant mood: {r.dominant_mood}')
    st.subheader('Overview')
    st.write(r.overview) if isinstance(r.overview,str) and r.overview.strip() else st.caption('Synopsis unavailable.')
    show_trailer(r)
    if 'personal_score' in r and pd.notna(r.personal_score):
        st.subheader('Why this movie matches you')
        for label, key, weight in [('Content similarity','personal_content',50),('Mood compatibility','personal_mood',25),('Genre affinity','personal_genre',15),('Quality','personal_quality',10)]:
            st.progress(float(r[key]), text=f'{label}: {r[key]*100:.1f}/100 · Weight {weight}%')
        st.info('Based on your liked and saved movies. Passed movies are excluded.')
    elif 'match_score' in r and pd.notna(r.match_score):
        st.subheader('Why this movie matches you')
        for label,score,weight in [('Content similarity','content_score','content_weight'),('Mood compatibility','mood_score','mood_weight'),('Quality score','quality_score','quality_weight')]:
            if r[weight]>0 and pd.notna(r[score]):
                st.write(f'**{label}: {r[score]*100:.1f}/100** · Weight {r[weight]*100:.1f}%')
                st.progress(float(np.clip(r[score],0,1)))
        st.info(e.explain_recommendation(r,mood=mood,reference_title=None if query else reference).replace('Text similarity to your query','Combined reference and query similarity' if reference and query else 'Text similarity to your query'))
    if 'sentiment_similarity' in r: st.caption(f'Plot sentiment similarity: {r.sentiment_similarity:.3f}')
    st.subheader('Emotional profile')
    for profile in e.MOOD_PROFILES:
        value=r.get(f'mood_{profile}')
        if pd.notna(value):
            st.progress(float(np.clip(value,0,1)),text=f'{profile}: {value*100:.1f}/100')
    st.caption('Estimated from plot metadata; not guaranteed viewer reactions.')
    movie_id=int(r.id)
    if st.button('Remove from Watchlist' if movie_id in st.session_state.watchlist else '♡ Add to my Watchlist',key=f'dialog-save-{movie_id}',type='primary',use_container_width=True):
        toggle_watchlist(movie_id)
        st.rerun()

MOOD_STYLE={
    'Feel-good':('😊','#59421C','Feel-good stories'),
    'Romantic':('💕','#58243F','Fall in love'),
    'Funny':('😂','#5A5023','Laugh out loud'),
    'Exciting':('🔥','#623323','Feel the adrenaline'),
    'Dark & Suspenseful':('🌙','#482745','Embrace the unknown'),
    'Emotional':('🥺','#293A60','Feel something deep'),
    'Inspiring':('✨','#4C3565','Discover new perspectives'),
    'Thought-provoking':('🧠','#343160','Challenge your mind'),
    None:('🎬','#30283F','Explore every feeling')}

def choose_mood(detected):
    key=f'cine-mood-{page}'
    previous=f'{key}-detected'
    if key not in st.session_state or st.session_state.get(previous)!=detected:
        st.session_state[key]=detected if detected in e.MOOD_PROFILES else (e.MOOD_PROFILES[0] if page=='Mood Match' else None)
    st.session_state[previous]=detected
    st.subheader('Your desired mood')
    if page=='Mood Match': st.caption('Every feeling has a story. Choose how you want to feel tonight.')
    choices=list(e.MOOD_PROFILES) if page=='Mood Match' else [None]+list(e.MOOD_PROFILES)
    for start in range(0,len(choices),3):
        cols=st.columns(3)
        for col,profile in zip(cols,choices[start:start+3]):
            emoji,color,description=MOOD_STYLE.get(profile,('✨','#30283F','Explore this mood'))
            selected=st.session_state[key]==profile
            with col:
                with st.container(border=True):
                    st.markdown(f'<div style="background:{color};border:2px solid {"#d8b4fe" if selected else color};border-radius:12px;padding:12px;text-align:center;color:white"><span style="font-size:28px">{emoji}</span><br>{escape(description)}</div>',unsafe_allow_html=True)
                    st.button(('✓ ' if selected else '')+(profile or 'Any'),key=f'{key}-{start}-{profile}',type='primary' if selected else 'secondary',use_container_width=True,on_click=set_mood,args=(key,profile))
    return st.session_state[key]

def set_mood(key,profile):
    st.session_state[key]=profile

def cards(rows, mood=None, reference=None, query=''):
    if rows.empty:
        st.info('No movies match. Try relaxing a filter.'); return
    movies = list(rows.iterrows())
    for start in range(0, len(movies), 3):
        cols = st.columns(3, gap='medium')
        for offset, (_, r) in enumerate(movies[start:start+3]):
            j = start + offset
            movie_id = int(r.id)
            with cols[offset]:
                with st.container(border=True):
                    if page != 'My Watchlist':
                        st.caption(f'TOP PICK · {j+1:02d}')
                    if isinstance(r.poster_url, str) and r.poster_url.startswith('https://'):
                        from html import escape
                        st.markdown(f'<img class="cine-poster" src="{escape(r.poster_url, quote=True)}" alt="{escape(str(r.title), quote=True)} poster" loading="lazy">', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="cine-poster cine-placeholder">🎬<br>Poster unavailable</div>', unsafe_allow_html=True)
                    st.markdown(f'**{r.title} ({r.year})**')
                    st.caption(f'★ {r.vote_average:.1f}/10 · {r.runtime:g} min')
                    if 'match_score' in r and pd.notna(r.match_score):
                        st.markdown(f':green[**{r.match_score:.1f}/100 Match index**]')
                        st.progress(float(np.clip(r.match_score/100,0,1)))
                    st.caption(' · '.join(r.genres))
                    if st.button('View details', key=f'details-{page}-{j}-{movie_id}', use_container_width=True):
                        movie_details(r, mood, reference, query)
                    st.button('✓ Saved · Remove' if movie_id in st.session_state.watchlist else '＋ Watchlist', key=f'{page}-{j}-{movie_id}', on_click=toggle_watchlist, args=(movie_id,), use_container_width=True)

if page=='Mood Match':
    mood=choose_mood(None)
    rows=df.copy()
    rows['content_score']=0.
    rows['mood_score']=rows[f'mood_{mood}'].fillna(0) if mood else 0.
    rows['quality_score']=rows.weighted_rating.fillna(0)/10
    rows['content_weight']=0.
    rows['mood_weight']=1. if mood else 0.
    rows['quality_weight']=0. if mood else 1.
    rows['hybrid_score']=rows.mood_score if mood else rows.quality_score
    rows['match_score']=rows.hybrid_score*100
    rows=rows.sort_values(['hybrid_score','weighted_rating'],ascending=False).head(12)
    st.divider()
    st.header(f'Your {mood} collection' if mood else 'Top picks for you')
    st.caption('Ranked by emotional compatibility' if mood else 'Ranked by weighted rating')
    cards(rows,mood)
elif page in ['Discover','Similar Movies']:
    query=st.text_input('Describe your ideal movie',placeholder='Funny romantic movie under 120 minutes, without horror') if page=='Discover' else ''
    prefs=e.parse_query(query); detected=prefs.pop('mood')
    mood=choose_mood(detected) if page=='Discover' else None
    ref=st.selectbox('Reference movie',[None]+list(labels),format_func=lambda x:x or 'None',index=1 if page=='Similar Movies' else 0)
    with st.expander('Advanced filters',expanded=False):
        c,d=st.columns(2)
        with c:
            years=st.slider('Year range',int(df.year.min()),int(df.year.max()),prefs['year_range'] or (int(df.year.min()),int(df.year.max())))
            period=st.checkbox('Filter exact release period')
            dates=st.date_input('Release dates',value=(df.release_date.min().date(),df.release_date.max().date())) if period else None
            rating=st.number_input('Minimum rating',0.,10.,float(min(prefs['min_rating'],10)),step=.1,format='%.2f')
            votes=st.number_input('Minimum votes',0,1000000,0)
            runtime=st.slider('Runtime (minutes)',1,max(600,int(df.runtime.max())),prefs['runtime_range'] or (1,max(600,int(df.runtime.max()))))
        with d:
            genres=st.multiselect('Genres',options('genres'),default=prefs['genres'])
            op=st.radio('Genre logic',['AND','OR'],index=int(prefs['genre_operator']=='OR'),horizontal=True)
            excluded=st.multiselect('Exclude genres',options('genres'),default=prefs['excluded_genres'])
            cert=st.multiselect('Certification',sorted(df.certification.dropna().unique()))
            langs=st.multiselect('Original language',sorted(df.original_language.dropna().unique()))
            actors=st.multiselect('Actors',options('cast'))
            directors=st.multiselect('Directors',options('directors'))
    filters=dict(year_range=years,min_rating=rating,min_votes=votes,runtime_range=runtime,genres=genres,genre_operator=op,excluded_genres=excluded,certifications=cert,languages=langs,actors=actors,directors=directors)
    if dates and len(dates)==2: filters['release_period']=dates
    method=st.radio('Recommendation method',['Hybrid','Content-Based','Sentiment-Based','Popular'],horizontal=True)
    if mood and method!='Hybrid': st.caption('The selected mood affects Hybrid recommendations only.')
    with st.expander('Personalize hybrid weights'):
        if page=='Discover':
            weights=tuple(st.slider(k,0,100,v,key=k) for k,v in [('Content',50),('Mood',35),('Quality',15)])
        else:
            weights=(st.slider('Content',0,100,50,key='similar-content'),0,st.slider('Quality',0,100,15,key='similar-quality'))
    if query: st.caption('Detected preferences are reflected in the controls. Review before searching.')
    candidates=e.filter_movies(**filters)
    st.caption(f'{len(candidates)} eligible movies')
    reference=df.loc[labels[ref],'title'] if ref else None
    if method=='Popular': rows=candidates.nlargest(12,'popularity')
    elif method=='Sentiment-Based':
        target=st.slider('Target plot sentiment',-1.,1.,float(df.loc[labels[ref],'sentiment_compound']) if ref else 0.8)
        rows=candidates.assign(sentiment_similarity=1-(candidates.sentiment_compound-target).abs()/2).sort_values(['sentiment_similarity','weighted_rating'],ascending=False).head(12)
    else:
        # Unique selected ID controls reference alignment, including remakes.
        if ref:
            idx=labels[ref]; scores=e.similarity_matrix[idx].copy()
        else: idx=None; scores=np.zeros(len(df))
        if query:
            q=e.tfidf.transform([query]); qs=cosine_similarity(q,e.tfidf_matrix).ravel(); scores=.6*scores+.4*qs if ref else qs
        w=np.array(weights if method=='Hybrid' else (1,0,0),float)
        if not ref and not query: w[0]=0
        if not mood: w[1]=0
        if not w.sum(): w[2]=1
        w/=w.sum()
        rows=candidates.drop(index=idx,errors='ignore').copy() if idx is not None else candidates.copy()
        rows['content_score']=scores[rows.index]; rows['mood_score']=rows[f'mood_{mood}'] if mood else 0.; rows['quality_score']=rows.weighted_rating.fillna(0)/10
        for k,v in zip(['content_weight','mood_weight','quality_weight'],w): rows[k]=v
        rows['hybrid_score']=rows.content_score*w[0]+rows.mood_score*w[1]+rows.quality_score*w[2]; rows['match_score']=rows.hybrid_score*100
        rows=rows.nlargest(12,'hybrid_score')
    st.divider()
    st.header('Top picks for you')
    st.caption(f'Showing {len(rows)} of {len(candidates)} eligible movies • {method}')
    cards(rows,mood,reference,query)
elif page in ['Surprise Me','Hidden Gems','Top Rated']:
    prefix = 'collection-' + page
    c1,c2,c3 = st.columns(3)
    with c1:
        genre = st.selectbox('Genre', [None]+options('genres'), format_func=lambda x:x or 'Any', key=prefix+'-genre')
    with c2:
        mood = st.selectbox('Desired mood', [None]+list(e.MOOD_PROFILES), format_func=lambda x:x or 'Any', key=prefix+'-mood')
    with c3:
        years = st.slider('Year range', int(df.year.min()), int(df.year.max()), (int(df.year.min()),int(df.year.max())), key=prefix+'-years')
    candidates = df[df.year.between(*years)].copy()
    if genre:
        candidates = candidates[candidates.genres.apply(lambda gs: genre in gs)]
    if mood:
        candidates = candidates[candidates.dominant_mood.eq(mood)]
    if page == 'Hidden Gems':
        quality_cut = df.weighted_rating.quantile(.75)
        popularity_cut = df.popularity.median()
        votes_cut = df.vote_count.median()
        candidates = candidates[(candidates.weighted_rating >= quality_cut) & (candidates.popularity <= popularity_cut) & (candidates.vote_count >= votes_cut)]
        st.caption('Top quartile of weighted ratings, at or below median popularity, and at least median vote count in the full catalog.')
    if page == 'Surprise Me':
        st.subheader('Find your movie match')
        st.caption('One movie. One decision. Endless stories.')
        state = st.session_state
        state.setdefault('swipe_likes', [])
        state.setdefault('swipe_passes', [])
        eligible = set(candidates.id.astype(int))
        seen = set(state.swipe_likes + state.swipe_passes)
        pool = candidates[~candidates.id.isin(seen)]
        current = state.get('swipe_current')
        if current not in eligible or current in seen:
            current = int(pool.sample(1).iloc[0].id) if not pool.empty else None
            state.swipe_current = current
        left, center, right = st.columns([1, 2, 1])
        with center:
            st.caption(f'♥ {len(state.swipe_likes)} liked · {len(state.swipe_passes)} passed · {len(pool)} remaining')
            if current is not None:
                r = candidates[candidates.id.eq(current)].iloc[0]
                with st.container(border=True):
                    if isinstance(r.poster_url, str) and r.poster_url.startswith('https://'):
                        st.markdown(f'<img src="{escape(r.poster_url, quote=True)}" alt="{escape(str(r.title), quote=True)} poster" style="display:block;width:100%;max-height:480px;object-fit:contain;border-radius:16px">', unsafe_allow_html=True)
                    else:
                        st.caption('Poster unavailable')
                    st.subheader(f'{r.title} ({r.year})')
                    st.caption(f'★ {r.vote_average:.1f}/10 · {r.runtime:g} min')
                    st.caption(' · '.join(r.genres))
                    x, info, heart = st.columns(3)
                    if x.button('❌ Pass', key='swipe-pass', use_container_width=True):
                        state.swipe_passes.append(current)
                        state.swipe_current = None
                        st.rerun()
                    if info.button('ⓘ Details', key='swipe-details', use_container_width=True):
                        movie_details(r, mood)
                    if heart.button('💜 Like', key='swipe-like', type='primary', use_container_width=True):
                        state.swipe_likes.append(current)
                        if current not in state.watchlist:
                            state.watchlist.append(current)
                        state.swipe_current = None
                        st.rerun()
            elif candidates.empty:
                st.info('No movies match. Try relaxing a filter.')
            else:
                st.success('You have explored every eligible movie!')
            if st.button('↻ Reset discovery', key='swipe-reset', use_container_width=True):
                state.swipe_likes = []
                state.swipe_passes = []
                state.swipe_current = None
                st.rerun()
        if state.swipe_likes:
            st.subheader('♡ Your movie matches')
            st.caption('Liked movies are also saved to your Watchlist.')
            matches = df[df.id.isin(state.swipe_likes)].copy()
            cards(matches)
    else:
        st.caption(f'{len(candidates)} eligible movies · Ranked by weighted rating')
        rows = candidates.sort_values(['weighted_rating','vote_count'], ascending=False).head(12)
        cards(rows, mood)
elif page=='For You':
    from collections import Counter
    state = st.session_state
    likes = set(state.get('swipe_likes', []))
    passes = set(state.get('swipe_passes', []))
    saved_ids = set(state.watchlist)
    positive = (likes | saved_ids) - passes
    favorites = df[df.id.isin(positive)]
    available = df[~df.id.isin(likes | saved_ids | passes)].copy()
    st.subheader('🧬 Your Cinema DNA')
    if len(favorites) < 3:
        st.info('Like or save at least three different movies to unlock your personalized recommendations. Visit Surprise Me or Discover to get started.')
        st.progress(min(len(favorites)/3, 1.), text=f'{len(favorites)} / 3 favorite movies')
        st.subheader('Popular starting points')
        cards(available.sort_values(['weighted_rating','vote_count'], ascending=False).head(12))
    else:
        genre_counts = Counter(g for genres in favorites.genres for g in set(genres))
        mood_cols = [f'mood_{m}' for m in e.MOOD_PROFILES]
        mood_matrix = df[mood_cols].fillna(0).clip(0,1).to_numpy(dtype=float)
        profile = favorites[mood_cols].fillna(0).clip(0,1).mean().to_numpy(dtype=float)
        with st.container(border=True):
            a,b,c = st.columns(3)
            a.metric('Favorite genre', genre_counts.most_common(1)[0][0] if genre_counts else 'Unknown')
            b.metric('Favorite mood', e.MOOD_PROFILES[int(np.argmax(profile))] if profile.max()>0 else 'Unknown')
            c.metric('Favorite movies', len(favorites))
            st.caption(f'{len(likes)} likes · {len(saved_ids)} saved · {len(passes)} passed (likes and saves deduplicated)')
            st.markdown('**Your genre preferences**')
            for genre,count in genre_counts.most_common(6):
                st.progress(count/len(favorites), text=f'{genre}: {count/len(favorites):.0%} of favorites')
            with st.expander('Your emotional profile'):
                for mood_name,value in zip(e.MOOD_PROFILES,profile):
                    st.progress(float(value), text=f'{mood_name}: {value*100:.1f}/100')
        content = np.clip(e.similarity_matrix[favorites.index].mean(axis=0),0,1)
        emotional = cosine_similarity(mood_matrix, profile.reshape(1,-1)).ravel()
        affinity = {g:n/len(favorites) for g,n in genre_counts.items()}
        genre_scores = df.genres.apply(lambda gs: np.mean([affinity.get(g,0.) for g in set(gs)]) if len(gs) else 0.)
        available['personal_content'] = content[available.index]
        available['personal_mood'] = np.clip(emotional[available.index],0,1)
        available['personal_genre'] = genre_scores.loc[available.index]
        available['personal_quality'] = (available.weighted_rating.fillna(0)/10).clip(0,1)
        available['personal_score'] = .50*available.personal_content + .25*available.personal_mood + .15*available.personal_genre + .10*available.personal_quality
        available['match_score'] = available.personal_score*100
        st.subheader('Recommended for you')
        st.caption('Personalized hybrid ranking · Match index is not a probability.')
        cards(available.sort_values(['personal_score','weighted_rating'],ascending=False).head(12))
    st.caption('Your profile updates during this session. Passed movies are excluded, not used as negative training examples.')
elif page=='My Watchlist':
    st.caption('Saved for this browser session.')
    saved=df[df.id.isin(st.session_state.watchlist)]
    if saved.empty:
        st.info('Your watchlist is empty. Add movies from Discover, Mood Match or Similar Movies.')
    else:
        cards(saved)
    if not saved.empty:
        st.header('Based on your watchlist')
        scores=e.similarity_matrix[saved.index].mean(axis=0)
        rec=df.assign(profile_similarity=scores).drop(index=saved.index).nlargest(6,'profile_similarity')
        st.dataframe(rec[['title','year','profile_similarity','vote_average']],hide_index=True)
elif page=='Compare':
    chosen=st.multiselect('Choose up to four movies',list(labels),max_selections=4)
    if chosen:
        rows=df.loc[[labels[x] for x in chosen]]
        st.dataframe(rows[['title','year','runtime','genres','vote_average','vote_count','budget','revenue','dominant_mood']],hide_index=True)
        st.plotly_chart(px.bar(rows,x='title',y='vote_average',title='TMDB ratings'))
        st.caption('Missing budget or revenue is not treated as zero.')
else:
    k1,k2,k3=st.columns(3)
    k1.metric('Movies',f'{len(df):,}')
    k2.metric('Average TMDB rating',f'{df.vote_average.mean():.2f}/10')
    k3.metric('Release years',f'{int(df.year.min())}–{int(df.year.max())}')
    st.header('Model benchmark')
    ev=pd.read_csv(A/'model_evaluation.csv'); st.dataframe(ev,hide_index=True)
    st.plotly_chart(px.bar(ev.melt(id_vars='Model',value_vars=['Genre Relevance','Mood Compatibility','Diversity']),x='variable',y='value',color='Model',barmode='group',template='plotly_dark',color_discrete_sequence=['#818CF8','#A855F7','#2DD4BF','#FBBF24'],title='Recommendation model comparison'))
    st.caption('Original notebook benchmark: six reference movies; proxy metrics, not user satisfaction.')
    st.plotly_chart(px.histogram(df,x='dominant_mood',title='Emotional profiles',template='plotly_dark',color_discrete_sequence=['#A855F7']))
    st.plotly_chart(px.histogram(df,x='year',title='Release year distribution',template='plotly_dark',color_discrete_sequence=['#818CF8']))
    st.header('Quality assurance'); st.dataframe(pd.read_csv(A/'quality_report.csv'),hide_index=True)
    st.dataframe(pd.read_csv(A/'data_quality.csv'),hide_index=True)
st.divider(); st.caption('This product uses the TMDB API but is not endorsed or certified by TMDB. Mood scores are estimates from plot metadata, not guaranteed viewer reactions.')
