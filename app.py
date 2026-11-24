                    else:
                        st.markdown('<div class="cine-poster cine-placeholder">🎬<br>Poster unavailable</div>', unsafe_allow_html=True)
                    st.markdown(f'**{r.title} ({r.year})**')
                    st.caption(f'★ {r.vote_average:.1f}/10 · {r.runtime:g} min')
                    if 'match_score' in r and pd.notna(r.match_score):
                        st.markdown(f':green[**{r.match_score:.1f}/100 Match index**]')
                        st.progress(float(np.clip(r.match_score/100,0,1)))
                    st.caption(' · '.join(r.genres))
                    with st.expander('View details'):
                        meta = [str(x) for x in (r.certification, r.original_language) if pd.notna(x) and str(x).strip().lower() not in ('none','nan','')]
                        if meta: st.caption(' · '.join(meta))
                        st.write(r.overview) if isinstance(r.overview,str) and r.overview.strip() else st.caption('Synopsis unavailable.')
                        if 'match_score' in r:
                            st.markdown('**Why this movie?**')
                            st.write(e.explain_recommendation(r,mood=mood,reference_title=None if query else reference).replace('Text similarity to your query','Combined reference and query similarity' if reference and query else 'Text similarity to your query'))
                        if 'sentiment_similarity' in r:
                            st.caption(f'Plot sentiment similarity: {r.sentiment_similarity:.3f}')
                    st.button('✓ Saved · Remove' if movie_id in st.session_state.watchlist else '＋ Watchlist', key=f'{page}-{j}-{movie_id}', on_click=toggle_watchlist, args=(movie_id,), use_container_width=True)

if page in ['Discover','Mood Match','Similar Movies']:
    query=st.text_input('Describe your ideal movie',placeholder='Funny romantic movie under 120 minutes, without horror') if page=='Discover' else ''
    prefs=e.parse_query(query); detected=prefs.pop('mood')
    mood=st.selectbox('Desired mood',[None]+e.MOOD_PROFILES,index=([None]+e.MOOD_PROFILES).index(detected) if detected in e.MOOD_PROFILES else (1 if page=='Mood Match' else 0),format_func=lambda x:x or 'Any')
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
    st.header('Top picks for you')
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
