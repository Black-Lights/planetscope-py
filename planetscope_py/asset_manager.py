#!/usr/bin/env python3
"""
PlanetScope-py Phase 4: Asset Management and Download System
Complete asset activation, download, and quota management with user controls.

This module implements comprehensive asset management capabilities including:
- Real-time quota monitoring using Analytics and Subscriptions APIs
- Interactive user confirmation workflows
- Parallel asset activation and download
- Progress tracking and error recovery
- ROI clipping integration with Orders API
"""

import asyncio
import logging
import time
import os
import json
import aiohttp
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
from shapely.geometry import Polygon, shape
from shapely.ops import transform
import pyproj

from .auth import PlanetAuth
from .rate_limiter import RateLimiter
from .exceptions import AssetError, ValidationError, RateLimitError
from .utils import calculate_area_km2, validate_geometry

logger = logging.getLogger(__name__)


class AssetStatus(Enum):
    """Asset activation and download status."""

    PENDING = "pending"
    ACTIVATING = "activating"
    ACTIVE = "active"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class QuotaInfo:
    """User quota information from Planet APIs."""

    current_usage_km2: float
    monthly_limit_km2: float
    remaining_km2: float
    usage_percentage: float
    download_estimate_km2: float
    estimated_scenes_count: int
    estimated_cost_usd: Optional[float] = None
    warning_threshold: float = 0.8  # 80% warning threshold

    @property
    def is_near_limit(self) -> bool:
        """Check if usage is near the monthly limit."""
        return self.usage_percentage >= self.warning_threshold

    @property
    def can_download(self) -> bool:
        """Check if download is possible within quota."""
        return (
            self.current_usage_km2 + self.download_estimate_km2
        ) <= self.monthly_limit_km2


@dataclass
class DownloadJob:
    """Individual asset download job tracking."""

    scene_id: str
    asset_type: str
    item_type: str = "PSScene"
    download_url: Optional[str] = None
    status: AssetStatus = AssetStatus.PENDING
    file_path: Optional[Path] = None
    file_size_mb: Optional[float] = None
    activation_time: Optional[datetime] = None
    download_start_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate total job duration."""
        if self.activation_time and self.completion_time:
            return (self.completion_time - self.activation_time).total_seconds()
        return None

    @property
    def is_expired(self) -> bool:
        """Check if download link has expired."""
        if self.status == AssetStatus.ACTIVE and self.activation_time:
            # Planet download links typically expire after 4 hours
            expiry_time = self.activation_time + timedelta(hours=4)
            return datetime.now() > expiry_time
        return False


class AssetManager:
    """
    Comprehensive asset activation, download, and quota management system.

    Provides intelligent quota monitoring, user confirmation workflows,
    parallel download management, and integration with Planet's various APIs.
    """

    def __init__(self, auth: PlanetAuth, config: Optional[Dict] = None):
        """Initialize asset manager.

        Args:
            auth: PlanetAuth instance for API authentication
            config: Configuration settings for download behavior
        """
        self.auth = auth
        self.session = auth.get_session()

        # API endpoints
        self.data_api_url = "https://api.planet.com/data/v1"
        self.analytics_api_url = "https://api.planet.com/analytics"
        self.subscriptions_api_url = "https://api.planet.com/subscriptions/v1"
        self.orders_api_url = "https://api.planet.com/orders/v2"

        # Configuration
        self.config = config or {}
        self.max_concurrent_downloads = self.config.get("max_concurrent_downloads", 5)
        self.default_asset_types = self.config.get("asset_types", ["ortho_analytic_4b"])
        self.chunk_size = self.config.get("download_chunk_size", 8192)  # 8KB chunks

        # Rate limiting
        rate_limits = self.config.get(
            "rate_limits",
            {
                "activation": 5,  # Planet limit: 5 activations per second
                "download": 15,  # Planet limit: 15 downloads per second
                "general": 10,  # General API limit: 10 requests per second
            },
        )
        self.rate_limiter = RateLimiter(rate_limits, self.session)

        # Progress tracking
        self.progress_callback: Optional[Callable] = None
        self.download_jobs: List[DownloadJob] = []

        logger.info("AssetManager initialized with Planet API integration")

    async def check_user_quota(self) -> QuotaInfo:
        """
        Check current user quota and usage from multiple Planet APIs.

        Returns:
            QuotaInfo with current usage statistics and limits

        Raises:
            AssetError: If quota information cannot be retrieved
        """
        try:
            # Try multiple approaches to get quota information
            quota_data = await self._get_quota_from_multiple_sources()

            current_usage = quota_data.get("current_usage_km2", 0.0)
            monthly_limit = quota_data.get(
                "monthly_limit_km2", 3000.0
            )  # Default 3000 km²

            remaining = max(0, monthly_limit - current_usage)
            usage_percentage = (
                min(1.0, current_usage / monthly_limit) if monthly_limit > 0 else 0.0
            )

            return QuotaInfo(
                current_usage_km2=current_usage,
                monthly_limit_km2=monthly_limit,
                remaining_km2=remaining,
                usage_percentage=usage_percentage,
                download_estimate_km2=0.0,  # Will be calculated separately
                estimated_scenes_count=0,
            )

        except Exception as e:
            logger.error(f"Failed to check user quota: {e}")
            return self._get_estimated_quota()

    async def _get_quota_from_multiple_sources(self) -> Dict:
        """Get quota information from multiple Planet API sources."""
        quota_data = {"current_usage_km2": 0.0, "monthly_limit_km2": 3000.0}

        # Method 1: Try Analytics API subscriptions
        try:
            analytics_quota = await self._get_quota_from_analytics_api()
            if analytics_quota:
                quota_data.update(analytics_quota)
                logger.info("Retrieved quota from Analytics API")
                return quota_data
        except Exception as e:
            logger.debug(f"Analytics API quota check failed: {e}")

        # Method 2: Try Subscriptions API
        try:
            subscriptions_quota = await self._get_quota_from_subscriptions_api()
            if subscriptions_quota:
                quota_data.update(subscriptions_quota)
                logger.info("Retrieved quota from Subscriptions API")
                return quota_data
        except Exception as e:
            logger.debug(f"Subscriptions API quota check failed: {e}")

        # Method 3: Try to infer from recent downloads (Data API)
        try:
            data_api_quota = await self._estimate_quota_from_data_api()
            if data_api_quota:
                quota_data.update(data_api_quota)
                logger.info("Estimated quota from Data API usage patterns")
                return quota_data
        except Exception as e:
            logger.debug(f"Data API quota estimation failed: {e}")

        logger.warning("Using default quota values - could not retrieve from any API")
        return quota_data

    async def _get_quota_from_analytics_api(self) -> Optional[Dict]:
        """Get quota information from Analytics API subscriptions."""
        try:
            url = f"{self.analytics_api_url}/subscriptions"
            response = self.rate_limiter.make_request("GET", url)

            if response.status_code == 200:
                subscriptions = response.json()

                # Find PlanetScope subscription
                for sub in subscriptions.get("subscriptions", []):
                    if "planetscope" in sub.get("plan_id", "").lower():
                        quota_info = sub.get("quota", {})
                        return {
                            "monthly_limit_km2": quota_info.get(
                                "area_limit_km2", 3000.0
                            ),
                            "current_usage_km2": quota_info.get("area_used_km2", 0.0),
                            "source": "analytics_api",
                        }

        except Exception as e:
            logger.debug(f"Analytics API quota check failed: {e}")

        return None

    async def _get_quota_from_subscriptions_api(self) -> Optional[Dict]:
        """Get quota information from Subscriptions API."""
        try:
            url = f"{self.subscriptions_api_url}/subscriptions"
            response = self.rate_limiter.make_request("GET", url)

            if response.status_code == 200:
                data = response.json()

                # Extract quota from active subscriptions
                for subscription in data.get("data", []):
                    if subscription.get("status") == "active":
                        quota = subscription.get("quota", {})
                        return {
                            "monthly_limit_km2": quota.get("limit", 3000.0),
                            "current_usage_km2": quota.get("used", 0.0),
                            "source": "subscriptions_api",
                        }

        except Exception as e:
            logger.debug(f"Subscriptions API quota check failed: {e}")

        return None

    async def _estimate_quota_from_data_api(self) -> Optional[Dict]:
        """Estimate quota usage from Data API patterns."""
        try:
            # Make a simple search request and check for quota information in headers
            search_url = f"{self.data_api_url}/quick-search"

            # Simple test search
            test_payload = {
                "item_types": ["PSScene"],
                "filter": {
                    "type": "DateRangeFilter",
                    "field_name": "acquired",
                    "config": {
                        "gte": "2025-01-01T00:00:00.000Z",
                        "lte": "2025-01-02T00:00:00.000Z",
                    },
                },
            }

            response = self.rate_limiter.make_request(
                "POST", search_url, json=test_payload
            )

            if response.status_code == 200:
                # Check response headers for quota information
                headers = response.headers
                if "X-RateLimit-Limit" in headers:
                    # Extract quota from rate limit headers if available
                    quota_limit = headers.get("X-Quota-Limit-SqKm")
                    quota_used = headers.get("X-Quota-Used-SqKm")

                    if quota_limit and quota_used:
                        return {
                            "monthly_limit_km2": float(quota_limit),
                            "current_usage_km2": float(quota_used),
                            "source": "response_headers",
                        }

        except Exception as e:
            logger.debug(f"Data API quota estimation failed: {e}")

        # If all methods fail, return None to use defaults
        return None

    def _get_estimated_quota(self) -> QuotaInfo:
        """Provide estimated quota information when actual data unavailable."""
        return QuotaInfo(
            current_usage_km2=0.0,
            monthly_limit_km2=3000.0,  # Standard account limit
            remaining_km2=3000.0,
            usage_percentage=0.0,
            download_estimate_km2=0.0,
            estimated_scenes_count=0,
        )

    # FIXED: Make this method synchronous to match test expectations
    def estimate_download_impact(self, scenes, roi=None, clip_to_roi=False):
        """
        Estimate download size and quota impact for scene collection.

        Args:
            scenes: List of Planet scene features
            roi: Region of interest for clipping estimation
            clip_to_roi: Whether scenes will be clipped to ROI

        Returns:
            QuotaInfo with download estimates added
        """
        # Run the async quota check in an event loop
        try:
            # Try to get current event loop
            try:
                loop = asyncio.get_running_loop()
                # If loop is running, create a new thread
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._estimate_download_impact_async(scenes, roi, clip_to_roi),
                    )
                    return future.result()
            except RuntimeError:
                # No loop running, use asyncio.run
                return asyncio.run(
                    self._estimate_download_impact_async(scenes, roi, clip_to_roi)
                )
        except Exception:
            # Fallback to sync implementation
            return self._estimate_download_impact_sync(scenes, roi, clip_to_roi)

    async def _estimate_download_impact_async(
        self, scenes, roi=None, clip_to_roi=False
    ):
        """Async implementation of download impact estimation."""
        total_area_km2 = 0.0
        scene_count = len(scenes)

        for scene in scenes:
            try:
                # Get scene geometry
                scene_geom = shape(scene["geometry"])

                if clip_to_roi and roi:
                    # Calculate clipped area
                    clipped_geom = scene_geom.intersection(roi)
                    area_km2 = calculate_area_km2(clipped_geom)
                else:
                    # Full scene area
                    area_km2 = calculate_area_km2(scene_geom)

                total_area_km2 += area_km2

            except Exception as e:
                logger.warning(f"Could not calculate area for scene: {e}")
                # Use average scene size as fallback (approximately 25 km²)
                total_area_km2 += 25.0

        # Get current quota
        current_quota = await self.check_user_quota()

        # Update with download estimates
        current_quota.download_estimate_km2 = total_area_km2
        current_quota.estimated_scenes_count = scene_count

        # Estimate cost (if pricing info available)
        if hasattr(self.config, "cost_per_km2"):
            current_quota.estimated_cost_usd = total_area_km2 * self.config.cost_per_km2

        return current_quota

    def _estimate_download_impact_sync(self, scenes, roi=None, clip_to_roi=False):
        """Synchronous fallback implementation."""
        total_area_km2 = 0.0
        scene_count = len(scenes)

        for scene in scenes:
            try:
                # Get scene geometry
                scene_geom = shape(scene["geometry"])

                if clip_to_roi and roi:
                    # Calculate clipped area
                    clipped_geom = scene_geom.intersection(roi)
                    area_km2 = calculate_area_km2(clipped_geom)
                else:
                    # Full scene area
                    area_km2 = calculate_area_km2(scene_geom)

                total_area_km2 += area_km2

            except Exception as e:
                logger.warning(f"Could not calculate area for scene: {e}")
                # Use average scene size as fallback (approximately 25 km²)
                total_area_km2 += 25.0

        # Get estimated quota (sync version)
        current_quota = self._get_estimated_quota()

        # Update with download estimates
        current_quota.download_estimate_km2 = total_area_km2
        current_quota.estimated_scenes_count = scene_count

        # Estimate cost (if pricing info available)
        if hasattr(self.config, "cost_per_km2"):
            current_quota.estimated_cost_usd = total_area_km2 * self.config.cost_per_km2

        return current_quota

    def get_user_confirmation(self, quota_info: QuotaInfo) -> bool:
        """
        Interactive user confirmation for downloads with quota display.

        Args:
            quota_info: QuotaInfo object with download estimates

        Returns:
            bool: True if user confirms download, False otherwise
        """
        print(f"\n" + "=" * 60)
        print(f"DOWNLOAD IMPACT ASSESSMENT")
        print(f"=" * 60)
        print(f"Scenes to download: {quota_info.estimated_scenes_count}")
        print(f"Estimated download size: {quota_info.download_estimate_km2:.2f} km²")
        print(
            f"Current quota usage: {quota_info.current_usage_km2:.2f} km² / {quota_info.monthly_limit_km2:.2f} km²"
        )
        print(f"Usage percentage: {quota_info.usage_percentage:.1%}")
        print(f"Remaining quota: {quota_info.remaining_km2:.2f} km²")

        if quota_info.estimated_cost_usd:
            print(f"Estimated cost: ${quota_info.estimated_cost_usd:.2f} USD")

        # Warnings
        if not quota_info.can_download:
            print(f"\n⚠️  WARNING: Download would exceed quota limit!")
            print(f"   Required: {quota_info.download_estimate_km2:.2f} km²")
            print(f"   Available: {quota_info.remaining_km2:.2f} km²")
            return False

        if quota_info.is_near_limit:
            print(
                f"\n⚠️  WARNING: Current usage is near limit ({quota_info.usage_percentage:.1%})"
            )

        print(f"=" * 60)

        # Get user confirmation
        while True:
            response = input(f"Proceed with download? (y/n): ").lower().strip()
            if response in ["y", "yes"]:
                return True
            elif response in ["n", "no"]:
                print("Download cancelled by user.")
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no.")

    async def activate_and_download_assets(
        self,
        scenes: List[Dict],
        asset_types: Optional[List[str]] = None,
        output_dir: str = "downloads",
        roi: Optional[Polygon] = None,
        clip_to_roi: bool = False,
        confirm_download: bool = True,
        max_concurrent: Optional[int] = None,
    ) -> List[DownloadJob]:
        """
        Main asset activation and download workflow with quota management.

        Args:
            scenes: List of Planet scene features
            asset_types: Asset types to download (default: ortho_analytic_4b)
            output_dir: Directory for downloaded files
            roi: Region of interest for clipping
            clip_to_roi: Whether to clip scenes to ROI
            confirm_download: Whether to ask for user confirmation
            max_concurrent: Maximum concurrent downloads

        Returns:
            List of DownloadJob objects with status and results
        """
        # Setup
        asset_types = asset_types or self.default_asset_types
        max_concurrent = max_concurrent or self.max_concurrent_downloads
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Check quota and get confirmation
        if confirm_download:
            quota_info = self.estimate_download_impact(scenes, roi, clip_to_roi)
            if not self.get_user_confirmation(quota_info):
                return []

        # Create download jobs
        self.download_jobs = []
        for scene in scenes:
            scene_id = scene["properties"]["id"]
            for asset_type in asset_types:
                job = DownloadJob(
                    scene_id=scene_id,
                    asset_type=asset_type,
                    item_type=scene["properties"].get("item_type", "PSScene"),
                )
                self.download_jobs.append(job)

        print(f"\nStarting download of {len(self.download_jobs)} assets...")
        print(f"Output directory: {output_path.absolute()}")

        # Phase 1: Asset Activation
        print(f"\nPhase 1: Activating {len(self.download_jobs)} assets...")
        await self._activate_assets_batch(self.download_jobs)

        # Phase 2: Wait for activation and download
        print(f"\nPhase 2: Monitoring activation and downloading...")
        await self._download_activated_assets(
            self.download_jobs, output_path, max_concurrent
        )

        # Summary
        self._print_download_summary(self.download_jobs)

        return self.download_jobs

    async def _activate_assets_batch(self, jobs: List[DownloadJob]) -> None:
        """Activate all assets in parallel with rate limiting."""

        async def activate_single_asset(job: DownloadJob) -> None:
            try:
                job.status = AssetStatus.ACTIVATING
                job.activation_time = datetime.now()

                # Get asset info
                assets_url = f"{self.data_api_url}/item-types/{job.item_type}/items/{job.scene_id}/assets"
                response = self.rate_limiter.make_request("GET", assets_url)

                if response.status_code == 200:
                    assets = response.json()

                    if job.asset_type in assets:
                        asset_info = assets[job.asset_type]

                        # Check if already active
                        if asset_info.get("status") == "active":
                            job.status = AssetStatus.ACTIVE
                            job.download_url = asset_info.get("location")
                            logger.info(
                                f"Asset {job.scene_id}:{job.asset_type} already active"
                            )
                            return

                        # Activate asset
                        activation_url = asset_info["_links"]["activate"]
                        activate_response = self.rate_limiter.make_request(
                            "GET", activation_url
                        )

                        if activate_response.status_code in [202, 204]:
                            logger.info(
                                f"Asset {job.scene_id}:{job.asset_type} activation requested"
                            )
                        else:
                            raise AssetError(
                                f"Activation failed: {activate_response.status_code}"
                            )
                    else:
                        raise AssetError(f"Asset type {job.asset_type} not available")
                else:
                    raise AssetError(f"Failed to get assets: {response.status_code}")

            except Exception as e:
                job.status = AssetStatus.FAILED
                job.error_message = str(e)
                logger.error(f"Failed to activate {job.scene_id}:{job.asset_type}: {e}")

        # Activate assets with rate limiting
        semaphore = asyncio.Semaphore(5)  # Limit concurrent activations

        async def activate_with_semaphore(job):
            async with semaphore:
                await activate_single_asset(job)

        # Execute activations
        tasks = [activate_with_semaphore(job) for job in jobs]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _download_activated_assets(
        self, jobs: List[DownloadJob], output_path: Path, max_concurrent: int
    ) -> None:
        """Monitor activation status and download assets when ready."""
        pending_jobs = [job for job in jobs if job.status != AssetStatus.FAILED]
        download_semaphore = asyncio.Semaphore(max_concurrent)

        async def monitor_and_download(job: DownloadJob) -> None:
            # Wait for activation
            max_wait_time = 300  # 5 minutes maximum wait
            start_wait = time.time()

            while job.status == AssetStatus.ACTIVATING:
                if time.time() - start_wait > max_wait_time:
                    job.status = AssetStatus.FAILED
                    job.error_message = "Activation timeout"
                    return

                try:
                    # Check activation status
                    assets_url = f"{self.data_api_url}/item-types/{job.item_type}/items/{job.scene_id}/assets"
                    response = self.rate_limiter.make_request("GET", assets_url)

                    if response.status_code == 200:
                        assets = response.json()
                        asset_info = assets.get(job.asset_type, {})

                        if asset_info.get("status") == "active":
                            job.status = AssetStatus.ACTIVE
                            job.download_url = asset_info.get("location")
                            break

                    await asyncio.sleep(5)  # Wait 5 seconds before checking again

                except Exception as e:
                    logger.warning(f"Error checking activation status: {e}")
                    await asyncio.sleep(5)

            # Download if active
            if job.status == AssetStatus.ACTIVE and job.download_url:
                async with download_semaphore:
                    await self._download_single_asset(job, output_path)

        # Monitor and download all assets
        tasks = [monitor_and_download(job) for job in pending_jobs]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _download_single_asset(self, job: DownloadJob, output_path: Path) -> None:
        """Download a single asset file."""
        try:
            job.status = AssetStatus.DOWNLOADING
            job.download_start_time = datetime.now()

            # Determine output filename
            filename = f"{job.scene_id}_{job.asset_type}.tif"
            file_path = output_path / filename
            job.file_path = file_path

            # Download with streaming
            async with aiohttp.ClientSession() as session:
                async with session.get(job.download_url) as response:
                    if response.status == 200:
                        total_size = int(response.headers.get("content-length", 0))
                        downloaded_size = 0

                        with open(file_path, "wb") as f:
                            async for chunk in response.content.iter_chunked(
                                self.chunk_size
                            ):
                                f.write(chunk)
                                downloaded_size += len(chunk)

                        job.file_size_mb = downloaded_size / (1024 * 1024)
                        job.status = AssetStatus.COMPLETED
                        job.completion_time = datetime.now()

                        logger.info(
                            f"Downloaded {job.scene_id}:{job.asset_type} ({job.file_size_mb:.1f} MB)"
                        )
                    else:
                        raise AssetError(f"Download failed: HTTP {response.status}")

        except Exception as e:
            job.status = AssetStatus.FAILED
            job.error_message = str(e)

            # Retry logic
            if job.retry_count < job.max_retries:
                job.retry_count += 1
                job.status = AssetStatus.PENDING
                logger.warning(
                    f"Retrying download {job.scene_id}:{job.asset_type} (attempt {job.retry_count})"
                )
                await asyncio.sleep(5)  # Wait before retry
                await self._download_single_asset(job, output_path)
            else:
                logger.error(f"Failed to download {job.scene_id}:{job.asset_type}: {e}")

    def _print_download_summary(self, jobs: List[DownloadJob]) -> None:
        """Print comprehensive download summary."""
        completed = [j for j in jobs if j.status == AssetStatus.COMPLETED]
        failed = [j for j in jobs if j.status == AssetStatus.FAILED]

        total_size_mb = sum(j.file_size_mb or 0 for j in completed)
        total_time = sum(
            j.duration_seconds or 0 for j in completed if j.duration_seconds
        )

        print(f"\n" + "=" * 60)
        print(f"DOWNLOAD SUMMARY")
        print(f"=" * 60)
        print(f"Completed: {len(completed)}/{len(jobs)} assets")
        print(f"Failed: {len(failed)} assets")
        print(f"Total Size: {total_size_mb:.1f} MB")

        if total_time > 0:
            avg_speed = total_size_mb / (total_time / 60)  # MB/minute
            print(f"Average Speed: {avg_speed:.1f} MB/min")

        if failed:
            print(f"\nFailed Downloads:")
            for job in failed:
                print(f"   • {job.scene_id}:{job.asset_type} - {job.error_message}")

        print(f"=" * 60)

    def get_download_status(self) -> Dict:
        """Get current download status summary."""
        if not self.download_jobs:
            return {"status": "no_jobs", "message": "No download jobs active"}

        status_counts = {}
        for status in AssetStatus:
            status_counts[status.value] = len(
                [j for j in self.download_jobs if j.status == status]
            )

        completed_jobs = [
            j for j in self.download_jobs if j.status == AssetStatus.COMPLETED
        ]
        total_size_mb = sum(j.file_size_mb or 0 for j in completed_jobs)

        return {
            "total_jobs": len(self.download_jobs),
            "status_breakdown": status_counts,
            "completed_size_mb": total_size_mb,
            "completed_files": [
                str(j.file_path) for j in completed_jobs if j.file_path
            ],
            "failed_jobs": [
                {
                    "scene_id": j.scene_id,
                    "asset_type": j.asset_type,
                    "error": j.error_message,
                }
                for j in self.download_jobs
                if j.status == AssetStatus.FAILED
            ],
        }

    def export_download_report(self, output_path: str) -> None:
        """Export detailed download report to JSON."""
        report = {
            "download_summary": self.get_download_status(),
            "job_details": [asdict(job) for job in self.download_jobs],
            "generated_at": datetime.now().isoformat(),
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"Download report saved to: {output_path}")


# Example usage and testing
if __name__ == "__main__":
    # This would typically be run from a notebook or script
    pass
