# ADR 0007 — Prefect (or a plain scheduled job), not Airflow

**Status:** Accepted · **Context:** §16

## Decision

Do not adopt Airflow. Use Prefect, or a plain scheduled job.

## Why

The ingestion DAG has about six nodes. Airflow brings a scheduler, a webserver, a metadata
database, and an executor — a footprint that will not coexist with Postgres, Redis, Phoenix,
Prometheus and Grafana inside a 16 GB laptop, which is the entire deployment target.

The honest reason people reach for Airflow here is to have it on a resume. That is a bad reason to
spend 2 GB of RAM.

## The claim being made

"I chose the simpler tool because the DAG has six nodes" is stronger engineering than adopting a
cluster scheduler to orchestrate six steps. Right-sizing infrastructure is a skill; over-adopting
is not.

## When to revisit

Cross-DAG dependencies, backfills over historical partitions, or a team needing multi-tenant
scheduling. None of those exist here.
