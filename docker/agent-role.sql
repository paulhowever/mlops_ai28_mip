SELECT 'CREATE ROLE dota_agent' WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'dota_agent') \gexec

ALTER ROLE dota_agent WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS CONNECTION LIMIT 10 PASSWORD :'password';
ALTER ROLE dota_agent SET default_transaction_read_only = on;
ALTER ROLE dota_agent SET statement_timeout = '10s';

SELECT format('GRANT CONNECT ON DATABASE %I TO dota_agent', current_database()) \gexec
GRANT USAGE ON SCHEMA public TO dota_agent;
SELECT 'GRANT SELECT ON TABLE public.tasks TO dota_agent' WHERE to_regclass('public.tasks') IS NOT NULL \gexec
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO dota_agent;
