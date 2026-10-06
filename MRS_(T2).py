# PAGE: Dataset / Research Mode
# ================================================================
elif page == "Dataset / Research Mode":
    st.title("🗂 Dataset / Research Mode")
    safety_banner()

    card_open()
    st.markdown("#### Music catalog")
    mode_badge("DEMO" if is_demo_dataset else "REAL DATA", "demo" if is_demo_dataset else "research")
    st.write(dataset_note)
    card_close()

    card_open()
    st.markdown("#### Trained recommendation models")
    if model_error:
        mode_badge("NOT LOADED", "warning")
        st.write(model_error)
        st.caption("Recommendations currently run in content-based fallback mode.")
    else:
        mode_badge("LOADED", "research")
        st.write("RNN and NCF trained weights loaded successfully.")
    card_close()

    card_open()
    st.markdown("#### WESAD (physiological research dataset)")
    try:
        subjects = physio.list_available_wesad_subjects()
    except Exception:
        subjects = []
    _wdf, _werr = _load_wesad_features()                                                     # [NEW]
    if _wdf is not None:
        mode_badge(f"PRECOMPUTED HRV FEATURES: {_wdf['subject'].nunique()} SUBJECTS", "research")
    if subjects:
        mode_badge(f"{len(subjects)} RAW SUBJECT(S) AVAILABLE", "research")
    elif _wdf is None:
        mode_badge("NOT AVAILABLE", "warning")
        st.write(f"Place downloaded subject folders at `{physio.WESAD_DIR}/S<id>/S<id>.pkl`. "
                 "WESAD requires registration at the official source (Schmidt et al., 2018).")
    card_close()

    card_open()
    st.markdown("#### Known dataset distinctions (do not merge blindly)")
    st.markdown("""
- **DEAM** — music emotion/audio-feature labels only. No physiological data.
- **WESAD** — physiological stress dataset (ECG/EDA/EMG/resp/temp/ACC). No music stimuli.
- **PMEmo** — music + emotion annotation + EDA (not HRV), song-level.
- **DEAP** — music-video stimuli + physiological signals; not directly comparable to WESAD's protocol.
""")
    card_close()

# ================================================================
# [NEW] PAGE: Backend Monitor (Admin) — for demos / presentations
# ================================================================
elif page == "Backend Monitor (Admin)":
    st.title("🛠 Backend Monitor (Admin)")
    admin_pw = _get_secret("ADMIN_PASSWORD")
    if not admin_pw:
        st.info("Set ADMIN_PASSWORD in Streamlit Secrets to unlock this page.")
    elif not st.session_state["admin_ok"]:
        pw = st.text_input("Admin password", type="password")
        if st.button("Unlock"):
            if hmac.compare_digest(str(pw), str(admin_pw)):
                st.session_state["admin_ok"] = True
                st.rerun()
            else:
                st.error("Wrong password.")
    else:
        if st.button("🔒 Lock"):
            st.session_state["admin_ok"] = False
            st.rerun()

        def _table(coll, sort_field, limit=100, drop=("_id", "qtable", "tipi", "dass", "whoqol")):
            proj = {f: 0 for f in drop}
            rows = list(coll.find({}, proj).sort(sort_field, -1).limit(limit))
            return pd.DataFrame(rows) if rows else pd.DataFrame()

        card_open()
        st.markdown("#### Collection sizes")
        names = ["users", "login_history", "user_profiles", "state_checkins", "recommendation_feedback",
                 "experiments", "physiological_measurements", "bias_assessments", "qtables"]
        st.dataframe(pd.DataFrame({"collection": names,
                                   "documents": [db.mongo_db[n].count_documents({}) for n in names]}),
                     use_container_width=True)
        card_close()

        card_open()
        st.markdown("#### Duplicate-user check")
        dups = list(db.users.aggregate([{"$group": {"_id": "$user", "n": {"$sum": 1}}}, {"$match": {"n": {"$gt": 1}}}]))
        pdups = list(db.profiles.aggregate([{"$group": {"_id": "$user", "n": {"$sum": 1}}}, {"$match": {"n": {"$gt": 1}}}]))
        if not dups and not pdups:
            st.success("No duplicate users: exactly one document per person in `users` and `user_profiles`.")
        else:
            st.error(f"Duplicates found — users: {dups}, profiles: {pdups}")
        card_close()

        card_open()
        st.markdown("#### Users (one row per person — login count, first/last login)")
        udf = _table(db.users, "last_login_utc")
        st.dataframe(udf, use_container_width=True) if not udf.empty else st.info("No users yet.")
        card_close()

        card_open()
        st.markdown("#### Check-in / check-out log (newest first)")
        ldf = _table(db.login_history, "login_time_utc")
        if ldf.empty:
            st.info("No logins yet.")
        else:
            st.dataframe(ldf, use_container_width=True)
            st.caption("logout_time_* stays empty if the person closed the tab without pressing Logout "
                       "or submitting the session feedback.")
        card_close()

        card_open()
        st.markdown("#### Short check-ins (returning users)")
        cdf = _table(db.state_checkins, "timestamp")
        st.dataframe(cdf, use_container_width=True) if not cdf.empty else st.info("None yet.")
        card_close()

        card_open()
        st.markdown("#### Song feedback (newest first)")
        fbdf = _table(db.recommendation_feedback, "timestamp")
        st.dataframe(fbdf, use_container_width=True) if not fbdf.empty else st.info("None yet.")
        card_close()