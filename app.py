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
<div class="cine-stats"><div><strong>{len(df):,}</strong><span>Movies</span></div><div><strong>AI</strong><span>Recommendations</span></div><div><strong>6</strong><span>Discovery tools</span></div></div>
</div></div>""", unsafe_allow_html=True)
st.session_state.setdefault('watchlist',[])
page=st.sidebar.radio('Explore',['Discover','Mood Match','Similar Movies','My Watchlist','Compare','Analytics'])
st.sidebar.caption(f'Powered by TMDB • {len(df):,} movies')
st.sidebar.caption(f'♡ {len(st.session_state.watchlist)} saved movies')
page_intro = {
    'Discover': ('Discover your next favorite', 'Describe what you want to watch or refine your preferences.'),
    'Mood Match': ('How do you want to feel tonight?', 'Every feeling has a story.'),
    'Similar Movies': ('More movies like your favorites', 'Select a reference movie to explore related stories.'),
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
    if 'match_score' in r and pd.notna(r.match_score):
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
    choices=[None]+list(e.MOOD_PROFILES)
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

if page in ['Discover','Mood Match','Similar Movies']:
    query=st.text_input('Describe your ideal movie',placeholder='Funny romantic movie under 120 minutes, without horror') if page=='Discover' else ''
    prefs=e.parse_query(query); detected=prefs.pop('mood')
    mood=choose_mood(detected)
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
        weights=tuple(st.slider(k,0,100,v,key=k) for k,v in [('Content',50),('Mood',35),('Quality',15)])
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
    st.header(f'Your {mood} collection' if page=='Mood Match' and mood and method=='Hybrid' and w[1]>0 else 'Top picks for you')
    st.caption(f'Showing {len(rows)} of {len(candidates)} eligible movies • {method}')
    cards(rows,mood,reference,query)
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
