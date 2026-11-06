"""Five-coordinate motor contract shared by actors and real-world adapters."""
import numpy as np

DRIVE, TURN, REACH, HEIGHT, GRIP = range(5)
ACTION_NAMES = ('drive', 'turn', 'reach', 'height', 'grip')


def validate_action(value):
    """Return an owned finite bounded action, without changing any live state."""
    try:
        action = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('action must be finite shape (5,) within [-1,1]') from exc
    if action.shape != (5,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
        raise ValueError('action must be finite shape (5,) within [-1,1]')
    return action.copy()


def govern_action(value, max_drive=1.0):
    """Enforce the drive governor after every source of action corrections."""
    action = validate_action(value)
    if not np.isscalar(max_drive) or not np.isfinite(max_drive) or not 0 <= max_drive <= 1:
        raise ValueError('max_drive must be finite within [0,1]')
    action[DRIVE] = np.clip(action[DRIVE], -max_drive, max_drive)
    return action
