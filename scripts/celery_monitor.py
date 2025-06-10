"""
Celery monitoring and management utilities
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from celery import Celery
from config.settings import settings

def get_celery_app():
    """Get configured Celery app"""
    return Celery(
        'trinity_tasks',
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend
    )

def check_celery_workers():
    """Check status of Celery workers"""
    app = get_celery_app()
    
    # Get active workers
    inspect = app.control.inspect()
    
    print("🔍 Celery Worker Status:")
    
    # Active workers
    active = inspect.active()
    if active:
        for worker, tasks in active.items():
            print(f"  {worker}: {len(tasks)} active tasks")
    else:
        print("  No active workers found")
    
    # Scheduled tasks
    scheduled = inspect.scheduled()
    if scheduled:
        for worker, tasks in scheduled.items():
            print(f"  {worker}: {len(tasks)} scheduled tasks")
    
    # Worker stats
    stats = inspect.stats()
    if stats:
        for worker, data in stats.items():
            print(f"  {worker}: {data.get('total', 0)} total tasks processed")

def trigger_cleanup():
    """Manually trigger cleanup task"""
    from scripts.celery_worker import cleanup_old_memories
    
    result = cleanup_old_memories.delay()
    print(f"🧹 Cleanup task triggered: {result.id}")
    return result

def trigger_backup():
    """Manually trigger backup task"""
    from scripts.celery_worker import backup_active_sessions
    
    result = backup_active_sessions.delay()
    print(f"Backup task triggered: {result.id}")
    return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Celery monitoring and management")
    parser.add_argument("--status", action="store_true", help="Check worker status")
    parser.add_argument("--cleanup", action="store_true", help="Trigger cleanup task")
    parser.add_argument("--backup", action="store_true", help="Trigger backup task")
    
    args = parser.parse_args()
    
    if args.status:
        check_celery_workers()
    elif args.cleanup:
        trigger_cleanup()
    elif args.backup:
        trigger_backup()
    else:
        parser.print_help()