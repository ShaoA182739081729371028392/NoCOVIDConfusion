# This File Handles the Logic of Basic Navigation
import streamlit as st
from streamlit.hashing import _CodeHasher
try:
    # Before Streamlit 0.65
    from streamlit.ReportThread import get_report_ctx
    from streamlit.server.Server import Server
except ModuleNotFoundError:
    # After Streamlit 0.65
    from streamlit.report_thread import get_report_ctx
    from streamlit.server.server import Server
import pages.home
import pages.ct
import pages.info
import pages.mask
import pages.profile
import pages.logged_in
from back_end.profile import DataBase
import time
cur_time = time.time()

GITHUB_LINK = 'https://github.com/ShaoA182739081729371028392/NoMoreCOVIDConfusion'
ICON = './pages/icon.png'
def main():
    state = _get_state()

    st.set_page_config(page_title = 'No More COVID Confusion!', page_icon=ICON)
    MENU = {
        'Home': pages.home,
        'CT': pages.ct,
        'Info': pages.info,
        'Mask': pages.mask,
        'Login': pages.profile
    }
    st.sidebar.title("No More COVID Confusion!")
    menu_selection = st.sidebar.radio('Menu', list(MENU))
    menu = MENU[menu_selection]
    profile = state.__getitem__('profile')
    with st.spinner(text = 'Loading'):
        if menu_selection == 'Login':
            if profile is not None:
                
                prev_time = state.__getitem__('time')
                state.__setitem__('time', cur_time)
                if prev_time is None:
                    pass
                else:
                    profile['quarantine'] = max(0, profile['quarantine'] - (cur_time - prev_time))
                state.__setitem__('profile', profile)
                profile = pages.logged_in.write(profile)
                
                if profile == 'logout':
                    state.__setitem__('profile', None)
                elif profile is not None:
                    state.__setitem__('profile', profile)
            else:
                profile = menu.write()
                if profile is not None:
                    state.__setitem__('profile', profile)
        else:
            menu.write()
    st.sidebar.info(f"Source Code can be Found [Here]({GITHUB_LINK})")
    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()

class _SessionState:
    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)
        
    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        if item == 'profile':
            # Save/Update the Profile
            if value is not None:
                DataBase.update_profile(value['_id'], quarantine = value['quarantine'], visited = value['visited'])

        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value
    
    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()
    
    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False
        
        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    
    session_info = Server.get_current()._get_session_info(session_id)
    
    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state
if __name__ == '__main__':
    main()