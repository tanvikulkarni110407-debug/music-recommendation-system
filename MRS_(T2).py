# ================================================================
# SIGN-IN / DASHBOARD
# ================================================================

if not st.session_state.verified:
    # Simple research-prototype sign-in.
    # OTP/Brevo has been removed; config.py no longer contains
    # BREVO_API_KEY, SENDER_EMAIL, or HOST_EMAILS.

    name_input = st.text_input("Enter your name or participant ID")
    email_input = st.text_input(
        "Email (optional, used only as an identifier)"
    )

    if st.button("Continue"):
        if name_input.strip():
            st.session_state.verified = True
            st.session_state.username = (
                name_input.strip().lower().replace(" ", "_")
            )
            st.session_state.user_email = (
                email_input.strip()
                or f"{st.session_state.username}@local"
            )

            ist_now = datetime.now(ZoneInfo("Asia/Kolkata"))

            db.login_history.insert_one({
                "user_email": st.session_state.user_email,
                "username": st.session_state.username,
                "login_time_ist": ist_now.strftime(
                    "%Y-%m-%d %I:%M:%S %p"
                ),
            })

            st.rerun()
        else:
            st.warning("Please enter a name or ID.")

else:
    st.success(
        f"Signed in as **{st.session_state.username}**"
    )

    if st.button("🚪 Logout"):
        for k in [
            "verified",
            "username",
            "user_email",
            "profile_doc",
            "profile_user",
        ]:
            st.session_state[k] = (
                None if k != "verified" else False
            )
        st.rerun()

card_close()

if st.session_state.verified:
    st.markdown(
        "Use the sidebar to continue: **Profile → Psychological Assessment → "
        "Physiological Input → Music Preference & Recommendation**."
    )

# ================================================================
# Everything below requires sign-in
# ================================================================
elif not db:
    st.error(
        "MongoDB connection is required. Configure MONGO_URI "
        "in Streamlit Secrets."
    )
elif not st.session_state.verified:
    st.warning("Please sign in on the Dashboard page first.")
