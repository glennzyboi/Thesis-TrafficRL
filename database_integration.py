"""
Database Integration Module for D3QN Traffic Control System
Handles connection to Supabase PostgreSQL and data synchronization
"""

import os
import json
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import pandas as pd
from pathlib import Path

try:
    import asyncpg
    import psycopg2
    from psycopg2.extras import Json
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("⚠️  PostgreSQL libraries not installed. Database integration disabled.")
    print("   Install with: pip install asyncpg psycopg2-binary")

class DatabaseManager:
    """
    Manages database connections and data synchronization for D3QN training
    Supports both real-time logging and batch synchronization
    """
    
    def __init__(self, connection_config: Optional[Dict] = None, enable_db: bool = True):
        self.enable_db = enable_db and POSTGRES_AVAILABLE
        self.connection_config = connection_config or self._load_db_config()
        self.connection_pool = None
        self.sync_connection = None
        
        if self.enable_db:
            print("🗄️  Database integration enabled")
        else:
            print("📁 Database integration disabled - using local files only")
    
    def _load_db_config(self) -> Dict:
        """Load database configuration from environment or config file"""
        # Try environment variables first (production)
        if os.getenv('SUPABASE_URL'):
            return {
                'host': os.getenv('SUPABASE_HOST'),
                'port': int(os.getenv('SUPABASE_PORT', 5432)),
                'database': os.getenv('SUPABASE_DB'),
                'user': os.getenv('SUPABASE_USER'),
                'password': os.getenv('SUPABASE_PASSWORD'),
                'sslmode': 'require'
            }
        
        # Try config file (development)
        config_file = Path('database_config.json')
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        
        # Default development config
        return {
            'host': 'localhost',
            'port': 5432,
            'database': 'traffic_rl',
            'user': 'postgres',
            'password': 'password',
            'sslmode': 'prefer'
        }
    
    async def initialize_connection_pool(self):
        """Initialize async connection pool for high-performance logging"""
        if not self.enable_db:
            return
        
        try:
            self.connection_pool = await asyncpg.create_pool(
                **self.connection_config,
                min_size=2,
                max_size=10,
                command_timeout=30
            )
            print("✅ Database connection pool initialized")
        except Exception as e:
            print(f"❌ Failed to initialize database pool: {e}")
            self.enable_db = False
    
    def get_sync_connection(self):
        """Get synchronous connection for simple operations"""
        if not self.enable_db:
            return None
        
        try:
            if not self.sync_connection or self.sync_connection.closed:
                self.sync_connection = psycopg2.connect(**self.connection_config)
            return self.sync_connection
        except Exception as e:
            print(f"❌ Failed to get sync connection: {e}")
            return None
    
    async def create_experiment(self, experiment_name: str, config: Dict) -> Optional[str]:
        """Create new experiment in database"""
        if not self.enable_db or not self.connection_pool:
            return None
        
        query = """
        INSERT INTO experiments (experiment_name, config, status, created_at)
        VALUES ($1, $2, 'running', $3)
        ON CONFLICT (experiment_name) 
        DO UPDATE SET config = $2, status = 'running'
        RETURNING id::text
        """
        
        try:
            async with self.connection_pool.acquire() as conn:
                result = await conn.fetchval(
                    query, 
                    experiment_name, 
                    json.dumps(config),
                    datetime.now(timezone.utc)
                )
                print(f"✅ Experiment '{experiment_name}' created/updated: {result}")
                return result
        except Exception as e:
            print(f"❌ Failed to create experiment: {e}")
            return None
    
    async def log_training_episode(self, experiment_name: str, episode_data: Dict) -> bool:
        """Log training episode to database"""
        if not self.enable_db or not self.connection_pool:
            return False
        
        try:
            async with self.connection_pool.acquire() as conn:
                # Use the stored procedure for clean insertion
                result = await conn.fetchval(
                    "SELECT insert_training_episode($1, $2)",
                    experiment_name,
                    json.dumps(episode_data)
                )
                print(f"📊 Episode {episode_data.get('episode_number')} logged: {result}")
                return True
        except Exception as e:
            print(f"❌ Failed to log episode: {e}")
            return False
    
    async def log_training_step(self, experiment_id: str, step_data: Dict) -> bool:
        """Log training step to database (for interval-based logging)"""
        if not self.enable_db or not self.connection_pool:
            return False
        
        query = """
        INSERT INTO training_steps (
            experiment_id, episode_number, step_number,
            active_vehicles, queue_lengths, waiting_times, speeds,
            state_vector, action_taken, immediate_reward,
            intersection_metrics, simulation_time
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        ON CONFLICT (experiment_id, episode_number, step_number) DO NOTHING
        """
        
        try:
            async with self.connection_pool.acquire() as conn:
                await conn.execute(
                    query,
                    experiment_id,
                    step_data['episode_number'],
                    step_data['step_number'],
                    step_data.get('active_vehicles', 0),
                    json.dumps(step_data.get('queue_lengths', {})),
                    json.dumps(step_data.get('waiting_times', {})),
                    json.dumps(step_data.get('speeds', {})),
                    json.dumps(step_data.get('state_vector', [])),
                    step_data.get('action_taken', 0),
                    step_data.get('immediate_reward', 0.0),
                    json.dumps(step_data.get('intersection_metrics', {})),
                    step_data.get('simulation_time', 0)
                )
                return True
        except Exception as e:
            print(f"❌ Failed to log step: {e}")
            return False
    
    def sync_local_logs_to_db(self, logs_directory: str = "production_logs") -> Dict[str, int]:
        """Synchronize local log files to database"""
        if not self.enable_db:
            return {'episodes': 0, 'steps': 0, 'errors': 0}
        
        stats = {'episodes': 0, 'steps': 0, 'errors': 0}
        logs_path = Path(logs_directory)
        
        if not logs_path.exists():
            print(f"📁 Logs directory not found: {logs_directory}")
            return stats
        
        # Sync episode logs
        for episode_file in logs_path.glob("*_episodes.jsonl"):
            experiment_name = episode_file.stem.replace('_episodes', '')
            try:
                with open(episode_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            episode_data = json.loads(line)
                            success = asyncio.run(
                                self.log_training_episode(experiment_name, episode_data)
                            )
                            if success:
                                stats['episodes'] += 1
                            else:
                                stats['errors'] += 1
            except Exception as e:
                print(f"❌ Error syncing {episode_file}: {e}")
                stats['errors'] += 1
        
        print(f"📤 Sync complete: {stats['episodes']} episodes, {stats['errors']} errors")
        return stats
    
    def get_experiment_analytics(self, experiment_name: str) -> Optional[Dict]:
        """Get comprehensive analytics for an experiment"""
        conn = self.get_sync_connection()
        if not conn:
            return None
        
        query = """
        SELECT 
            e.experiment_name,
            e.status,
            e.total_episodes,
            e.best_reward,
            e.convergence_episode,
            e.created_at,
            AVG(te.total_reward) as avg_reward,
            STDDEV(te.total_reward) as reward_std,
            AVG(te.passenger_throughput) as avg_passenger_throughput,
            AVG(te.avg_waiting_time) as avg_waiting_time,
            COUNT(te.id) as episodes_logged
        FROM experiments e
        LEFT JOIN training_episodes te ON e.id = te.experiment_id
        WHERE e.experiment_name = %s
        GROUP BY e.id, e.experiment_name, e.status, e.total_episodes, 
                 e.best_reward, e.convergence_episode, e.created_at
        """
        
        try:
            cursor = conn.cursor()
            cursor.execute(query, (experiment_name,))
            result = cursor.fetchone()
            
            if result:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, result))
            return None
        except Exception as e:
            print(f"❌ Failed to get analytics: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def export_experiment_data(self, experiment_name: str, output_dir: str = "exports") -> bool:
        """Export experiment data for analysis"""
        conn = self.get_sync_connection()
        if not conn:
            return False
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        try:
            # Export episodes
            episodes_query = """
            SELECT te.* FROM training_episodes te
            JOIN experiments e ON te.experiment_id = e.id
            WHERE e.experiment_name = %s
            ORDER BY te.episode_number
            """
            
            episodes_df = pd.read_sql(episodes_query, conn, params=(experiment_name,))
            episodes_file = output_path / f"{experiment_name}_episodes.csv"
            episodes_df.to_csv(episodes_file, index=False)
            
            # Export steps (if any)
            steps_query = """
            SELECT ts.* FROM training_steps ts
            JOIN experiments e ON ts.experiment_id = e.id
            WHERE e.experiment_name = %s
            ORDER BY ts.episode_number, ts.step_number
            """
            
            steps_df = pd.read_sql(steps_query, conn, params=(experiment_name,))
            if not steps_df.empty:
                steps_file = output_path / f"{experiment_name}_steps.csv"
                steps_df.to_csv(steps_file, index=False)
            
            print(f"📤 Exported data to {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to export data: {e}")
            return False
        finally:
            if conn:
                conn.close()
    
    async def close_connections(self):
        """Clean up database connections"""
        if self.connection_pool:
            await self.connection_pool.close()
        if self.sync_connection and not self.sync_connection.closed:
            self.sync_connection.close()


class ProductionLoggerWithDB:
    """
    Enhanced production logger with database integration
    Maintains local files as backup and syncs to database
    """
    
    def __init__(self, experiment_name: str, log_interval: int = 10, enable_db: bool = True):
        self.experiment_name = experiment_name
        self.log_interval = log_interval
        self.db_manager = DatabaseManager(enable_db=enable_db)
        
        # Local logging (same as before)
        self.logs_dir = Path("production_logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        self.episode_file = self.logs_dir / f"{experiment_name}_episodes.jsonl"
        self.steps_file = self.logs_dir / f"{experiment_name}_steps.jsonl"
        self.summary_file = self.logs_dir / f"{experiment_name}_summary.json"
        
        self.step_buffer = []
        self.episode_count = 0
        self.experiment_id = None
    
    async def initialize(self, config: Dict):
        """Initialize database connections and experiment"""
        await self.db_manager.initialize_connection_pool()
        
        if self.db_manager.enable_db:
            self.experiment_id = await self.db_manager.create_experiment(
                self.experiment_name, config
            )
    
    async def log_episode(self, episode_data: Dict):
        """Log episode to both local file and database"""
        # Local logging
        with open(self.episode_file, 'a', encoding='utf-8') as f:
            json.dump(episode_data, f, ensure_ascii=False)
            f.write('\n')
        
        # Database logging
        if self.db_manager.enable_db:
            await self.db_manager.log_training_episode(
                self.experiment_name, episode_data
            )
        
        self.episode_count += 1
    
    async def log_step(self, step_data: Dict):
        """Buffer steps and log to database at intervals"""
        self.step_buffer.append(step_data)
        
        # Log to database at intervals
        if len(self.step_buffer) >= self.log_interval:
            await self._flush_steps()
    
    async def _flush_steps(self):
        """Flush step buffer to local file and database"""
        if not self.step_buffer:
            return
        
        # Local logging
        with open(self.steps_file, 'a', encoding='utf-8') as f:
            for step in self.step_buffer:
                json.dump(step, f, ensure_ascii=False)
                f.write('\n')
        
        # Database logging
        if self.db_manager.enable_db and self.experiment_id:
            for step in self.step_buffer:
                await self.db_manager.log_training_step(self.experiment_id, step)
        
        self.step_buffer.clear()
    
    async def finalize(self, summary_data: Dict):
        """Finalize logging and close connections"""
        # Flush remaining steps
        await self._flush_steps()
        
        # Save summary
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        # Close database connections
        await self.db_manager.close_connections()
        
        print(f"📊 Logging finalized: {self.episode_count} episodes")


def create_sample_config():
    """Create sample database configuration file"""
    sample_config = {
        "host": "your-supabase-host.supabase.co",
        "port": 5432,
        "database": "postgres",
        "user": "postgres",
        "password": "your-password",
        "sslmode": "require"
    }
    
    with open('database_config_sample.json', 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    print("📝 Sample database config created: database_config_sample.json")
    print("   Copy to database_config.json and update with your credentials")


if __name__ == "__main__":
    # Demo usage
    import asyncio
    
    async def demo():
        # Create sample config
        create_sample_config()
        
        # Demo logger (will use local files only without DB config)
        logger = ProductionLoggerWithDB("demo_experiment", enable_db=False)
        await logger.initialize({"learning_rate": 0.001})
        
        # Log sample episode
        await logger.log_episode({
            "episode_number": 1,
            "total_reward": 150.5,
            "passenger_throughput": 1200
        })
        
        await logger.finalize({"total_episodes": 1})
        print("✅ Demo completed")
    
    asyncio.run(demo())
