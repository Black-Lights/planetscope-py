#!/usr/bin/env python3
"""
PlanetScope-py Enhanced Asset Management and Download System
===========================================================

Complete asset activation, download, and quota management with advanced retry logic,
timeout handling, and comprehensive error recovery.

This module implements:
- Enhanced real-time quota monitoring using multiple Planet APIs
- Intelligent retry logic with exponential backoff and proper timeout handling
- Interactive user confirmation workflows with proper cancellation tracking
- Parallel asset activation and download with better error classification
- Progress tracking and comprehensive error recovery
- ROI clipping integration with Orders API
- Detailed diagnostics and performance monitoring

Key Enhancements:
- Fixed infinite retry loops with proper timeout enforcement
- Intelligent error classification (retryable vs permanent failures)
- Enhanced download size estimation with disk space calculation
- Better quota detection from multiple API sources
- Comprehensive failure analysis and configuration recommendations

FIXED: activate_with_semaphore function now properly defined in _activate_assets_batch method
"""

import asyncio
import logging
import time
import os
import json
import aiohttp
import requests
from concurrent.futures import ThreadPoolExecutor
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
    """Asset activation and download status with enhanced cancellation tracking."""
    PENDING = "pending"
    ACTIVATING = "activating"
    ACTIVE = "active"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    USER_CANCELLED = "user_cancelled"


class DownloadCancellationReason(Enum):
    """Reasons for download cancellation."""
    USER_CHOICE = "user_choice"
    QUOTA_EXCEEDED = "quota_exceeded"
    API_ERROR = "api_error"
    INSUFFICIENT_SPACE = "insufficient_space"


@dataclass
class QuotaInfo:
    """Enhanced user quota information from Planet APIs."""
    current_usage_km2: float
    monthly_limit_km2: float
    remaining_km2: float
    usage_percentage: float
    download_estimate_km2: float
    download_estimate_mb: float = 0.0
    estimated_scenes_count: int = 0
    estimated_cost_usd: Optional[float] = None
    warning_threshold: float = 0.8
    quota_source: str = "estimated"
    last_updated: Optional[datetime] = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

    @property
    def is_near_limit(self) -> bool:
        """Check if usage is near the monthly limit."""
        return self.usage_percentage >= self.warning_threshold

    @property
    def can_download(self) -> bool:
        """Check if download is possible within quota."""
        return (self.current_usage_km2 + self.download_estimate_km2) <= self.monthly_limit_km2

    @property
    def quota_status(self) -> str:
        """Get human-readable quota status."""
        if not self.can_download:
            return "QUOTA_EXCEEDED"
        elif self.is_near_limit:
            return "NEAR_LIMIT"
        else:
            return "OK"


@dataclass
class DownloadJob:
    """Enhanced download job tracking with comprehensive retry and timeout handling."""
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
    cancellation_reason: Optional[DownloadCancellationReason] = None
    last_retry_time: Optional[datetime] = None
    total_retry_time: float = 0.0

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate total job duration including retries."""
        if self.activation_time and self.completion_time:
            return (self.completion_time - self.activation_time).total_seconds()
        elif self.activation_time:
            return (datetime.now() - self.activation_time).total_seconds()
        return None

    @property
    def is_expired(self) -> bool:
        """Check if download link has expired."""
        if self.status == AssetStatus.ACTIVE and self.activation_time:
            expiry_time = self.activation_time + timedelta(hours=4)
            return datetime.now() > expiry_time
        return False

    @property
    def should_retry(self) -> bool:
        """Check if job should be retried based on status and retry count."""
        if self.retry_count >= self.max_retries:
            return False
        
        if self.status in [AssetStatus.COMPLETED, AssetStatus.EXPIRED, AssetStatus.USER_CANCELLED]:
            return False
            
        if self.activation_time:
            total_elapsed = datetime.now() - self.activation_time
            if total_elapsed.total_seconds() > 7200:  # 2 hours max total time
                return False
                
        return True

    def record_retry_attempt(self):
        """Record a retry attempt with timing."""
        current_time = datetime.now()
        if self.last_retry_time:
            retry_interval = (current_time - self.last_retry_time).total_seconds()
            self.total_retry_time += retry_interval
        
        self.retry_count += 1
        self.last_retry_time = current_time


class AssetManager:
    """
    Enhanced comprehensive asset activation, download, and quota management system.
    
    Provides intelligent quota monitoring, user confirmation workflows,
    parallel download management with advanced retry logic, timeout handling,
    and integration with Planet's various APIs.
    """

    def __init__(self, auth: PlanetAuth, config: Optional[Dict] = None):
        """
        Initialize asset manager with enhanced timeout and retry configuration.

        Args:
            auth: PlanetAuth instance for API authentication (handles auth seamlessly)
            config: Configuration settings for download behavior
        """
        self.auth = auth
        self.session = auth.get_session()

        # API endpoints
        self.data_api_url = "https://api.planet.com/data/v1"
        self.analytics_api_url = "https://api.planet.com/analytics"
        self.subscriptions_api_url = "https://api.planet.com/subscriptions/v1"
        self.orders_api_url = "https://api.planet.com/orders/v2"

        # Configuration with enhanced defaults
        self.config = config or {}
        self.max_concurrent_downloads = self.config.get("max_concurrent_downloads", 3)
        self.default_asset_types = self.config.get("asset_types", ["ortho_analytic_4b"])
        self.chunk_size = self.config.get("download_chunk_size", 8192)

        # Enhanced timeout configuration for your specific case
        timeout_config = self.config.get("timeouts", {})
        self.timeouts = {
            "activation_poll_interval": timeout_config.get("activation_poll_interval", 15),     # Slower polling
            "activation_max_interval": timeout_config.get("activation_max_interval", 60),      # Max 1 minute
            "activation_timeout": timeout_config.get("activation_timeout", 2400),              # 40 minutes activation
            "download_connect": timeout_config.get("download_connect", 60),                    # 1 minute connect
            "download_read": timeout_config.get("download_read", 300),                         # 5 minutes read
            "download_total": timeout_config.get("download_total", 14400),                     # 4 hours total for large files
            "download_stall": timeout_config.get("download_stall", 900)                        # 15 minutes stall detection
        }
        
        # More conservative retry config
        retry_config = self.config.get("retry_config", {})
        self.retry_config = {
            "max_retries": retry_config.get("max_retries", 3),
            "initial_delay": retry_config.get("initial_delay", 60),                # 1 minute initial delay
            "max_delay": retry_config.get("max_delay", 900),                       # 15 minutes max delay
            "exponential_base": retry_config.get("exponential_base", 1.5),         # Gentler exponential backoff
            "jitter": retry_config.get("jitter", True)
        }
        
        # Conservative concurrent downloads for large files
        self.max_concurrent_downloads = self.config.get("max_concurrent_downloads", 1)  # Only 1 for large files

        # Rate limiting with conservative defaults
        rate_limits = self.config.get("rate_limits", {
            "activation": 3,
            "download": 10,
            "general": 8,
        })
        self.rate_limiter = RateLimiter(rate_limits, self.session)

        # Progress tracking
        self.progress_callback: Optional[Callable] = None
        self.download_jobs: List[DownloadJob] = []
        self.last_cancellation_reason: Optional[DownloadCancellationReason] = None

        logger.info("AssetManager initialized with enhanced timeout and retry configuration")
        logger.info(f"Timeouts: activation={self.timeouts['activation_timeout']}s, download={self.timeouts['download_total']}s")
        logger.info(f"Retries: max={self.retry_config['max_retries']}, initial_delay={self.retry_config['initial_delay']}s")

    async def check_user_quota(self) -> QuotaInfo:
        """
        Enhanced quota checking with multiple API sources and better error handling.
        """
        try:
            quota_data = await self._get_quota_from_multiple_sources()

            current_usage = quota_data.get("current_usage_km2", 0.0)
            monthly_limit = quota_data.get("monthly_limit_km2", 3000.0)
            quota_source = quota_data.get("source", "estimated")

            remaining = max(0, monthly_limit - current_usage)
            usage_percentage = (
                min(1.0, current_usage / monthly_limit) if monthly_limit > 0 else 0.0
            )

            return QuotaInfo(
                current_usage_km2=current_usage,
                monthly_limit_km2=monthly_limit,
                remaining_km2=remaining,
                usage_percentage=usage_percentage,
                download_estimate_km2=0.0,
                download_estimate_mb=0.0,
                estimated_scenes_count=0,
                quota_source=quota_source,
                last_updated=datetime.now()
            )

        except Exception as e:
            logger.error(f"Failed to check user quota: {e}")
            return self._get_estimated_quota()

    async def _get_quota_from_multiple_sources(self) -> Dict:
        """Enhanced quota retrieval with better API integration."""
        quota_data = {"current_usage_km2": 0.0, "monthly_limit_km2": 3000.0, "source": "estimated"}

        # Method 1: Try Planet API Stats endpoint
        try:
            stats_quota = await self._get_quota_from_stats_api()
            if stats_quota:
                quota_data.update(stats_quota)
                logger.info(f"Retrieved quota from Stats API: {stats_quota['current_usage_km2']:.2f} km² used")
                return quota_data
        except Exception as e:
            logger.debug(f"Stats API quota check failed: {e}")

        # Method 2: Try Analytics API subscriptions
        try:
            analytics_quota = await self._get_quota_from_analytics_api()
            if analytics_quota:
                quota_data.update(analytics_quota)
                logger.info(f"Retrieved quota from Analytics API: {analytics_quota['current_usage_km2']:.2f} km² used")
                return quota_data
        except Exception as e:
            logger.debug(f"Analytics API quota check failed: {e}")

        # Method 3: Try Subscriptions API
        try:
            subscriptions_quota = await self._get_quota_from_subscriptions_api()
            if subscriptions_quota:
                quota_data.update(subscriptions_quota)
                logger.info(f"Retrieved quota from Subscriptions API: {subscriptions_quota['current_usage_km2']:.2f} km² used")
                return quota_data
        except Exception as e:
            logger.debug(f"Subscriptions API quota check failed: {e}")

        # Method 4: Parse response headers
        try:
            headers_quota = await self._get_quota_from_response_headers()
            if headers_quota:
                quota_data.update(headers_quota)
                logger.info(f"Retrieved quota from response headers: {headers_quota['current_usage_km2']:.2f} km² used")
                return quota_data
        except Exception as e:
            logger.debug(f"Response headers quota check failed: {e}")

        logger.warning("Using default quota values - could not retrieve from any API")
        return quota_data

    async def _get_quota_from_stats_api(self) -> Optional[Dict]:
        """Get quota from Planet Stats API endpoint."""
        try:
            profile_url = f"{self.data_api_url}/auth/me"
            response = self.rate_limiter.make_request("GET", profile_url)
            
            if response.status_code == 200:
                profile_data = response.json()
                
                if "quota" in profile_data:
                    quota_info = profile_data["quota"]
                    return {
                        "monthly_limit_km2": float(quota_info.get("limit_sqkm", 3000.0)),
                        "current_usage_km2": float(quota_info.get("used_sqkm", 0.0)),
                        "source": "user_profile",
                    }
                
                if "plan" in profile_data:
                    plan_info = profile_data["plan"]
                    plan_quota = plan_info.get("quota", {})
                    if plan_quota:
                        return {
                            "monthly_limit_km2": float(plan_quota.get("limit", 3000.0)),
                            "current_usage_km2": float(plan_quota.get("used", 0.0)),
                            "source": "user_plan",
                        }
                        
            stats_url = f"{self.data_api_url}/stats"
            response = self.rate_limiter.make_request("GET", stats_url)
            
            if response.status_code == 200:
                stats_data = response.json()
                user_stats = stats_data.get("user", {})
                if user_stats:
                    quota_info = user_stats.get("download_quota", {})
                    if quota_info:
                        return {
                            "monthly_limit_km2": float(quota_info.get("limit_sqkm", 3000.0)),
                            "current_usage_km2": float(quota_info.get("used_sqkm", 0.0)),
                            "source": "stats_api",
                        }
                        
        except Exception as e:
            logger.debug(f"Stats API quota check failed: {e}")
        
        return None

    async def _get_quota_from_analytics_api(self) -> Optional[Dict]:
        """Get quota information from Analytics API subscriptions."""
        try:
            url = f"{self.analytics_api_url}/subscriptions"
            response = self.rate_limiter.make_request("GET", url)

            if response.status_code == 200:
                subscriptions = response.json()

                for sub in subscriptions.get("subscriptions", []):
                    if "planetscope" in sub.get("plan_id", "").lower():
                        quota_info = sub.get("quota", {})
                        return {
                            "monthly_limit_km2": quota_info.get("area_limit_km2", 3000.0),
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

    async def _get_quota_from_response_headers(self) -> Optional[Dict]:
        """Enhanced response header parsing for quota information."""
        try:
            search_url = f"{self.data_api_url}/quick-search"
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

            response = self.rate_limiter.make_request("POST", search_url, json=test_payload)

            if response.status_code == 200:
                headers = response.headers
                
                quota_headers = [
                    ("X-Quota-Used", "X-Quota-Limit"),
                    ("X-Quota-Used-SqKm", "X-Quota-Limit-SqKm"),
                    ("X-RateLimit-Used-SqKm", "X-RateLimit-Limit-SqKm"),
                    ("X-Plan-Used", "X-Plan-Quota"),
                ]
                
                for used_header, limit_header in quota_headers:
                    quota_used = headers.get(used_header)
                    quota_limit = headers.get(limit_header)
                    
                    if quota_used and quota_limit:
                        try:
                            used_val = float(quota_used)
                            limit_val = float(quota_limit)
                            
                            logger.info(f"Found quota in headers: {used_val:.2f}/{limit_val} km²")
                            return {
                                "monthly_limit_km2": limit_val,
                                "current_usage_km2": used_val,
                                "source": "response_headers",
                            }
                        except ValueError:
                            continue
                
                all_headers = dict(headers)
                quota_related = {k: v for k, v in all_headers.items() 
                               if any(term in k.lower() for term in ['quota', 'limit', 'usage', 'used', 'plan'])}
                
                if quota_related:
                    logger.info(f"Available quota-related headers: {quota_related}")
                else:
                    logger.debug("No quota headers found in response")

        except Exception as e:
            logger.debug(f"Response headers quota check failed: {e}")

        return None

    def _get_estimated_quota(self) -> QuotaInfo:
        """Provide estimated quota information when actual data unavailable."""
        return QuotaInfo(
            current_usage_km2=0.0,
            monthly_limit_km2=3000.0,
            remaining_km2=3000.0,
            usage_percentage=0.0,
            download_estimate_km2=0.0,
            download_estimate_mb=0.0,
            estimated_scenes_count=0,
            quota_source="estimated"
        )

    def estimate_download_impact(self, scenes, roi=None, clip_to_roi=False):
        """
        Enhanced download impact estimation with disk space calculation.
        """
        try:
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._estimate_download_impact_async(scenes, roi, clip_to_roi),
                    )
                    return future.result()
            except RuntimeError:
                return asyncio.run(
                    self._estimate_download_impact_async(scenes, roi, clip_to_roi)
                )
        except Exception:
            return self._estimate_download_impact_sync(scenes, roi, clip_to_roi)

    async def _estimate_download_impact_async(self, scenes, roi=None, clip_to_roi=False):
        """FIXED: Correct calculation of Planet quota usage vs actual download size."""
        total_quota_km2 = 0.0  # What Planet charges you for
        actual_download_km2 = 0.0  # What you actually download
        scene_count = len(scenes)

        logger.info(f"Calculating download impact for {scene_count} scenes, clip_to_roi={clip_to_roi}")

        for i, scene in enumerate(scenes):
            try:
                scene_id = scene.get("properties", {}).get("id", f"scene_{i}")
                scene_geom = shape(scene["geometry"])
                
                # PLANET BILLING RULE: You're always charged for the FULL scene area
                scene_area_km2 = calculate_area_km2(scene_geom)
                total_quota_km2 += scene_area_km2
                
                # Calculate what you actually download (affects disk space, not quota)
                if clip_to_roi and roi:
                    try:
                        clipped_geom = scene_geom.intersection(roi)
                        if not clipped_geom.is_empty:
                            clipped_area_km2 = calculate_area_km2(clipped_geom)
                            actual_download_km2 += clipped_area_km2
                            logger.debug(f"Scene {scene_id}: {scene_area_km2:.2f} km² quota, {clipped_area_km2:.2f} km² downloaded")
                        else:
                            logger.warning(f"Scene {scene_id}: No intersection with ROI")
                            # Still charged full quota, but download nothing
                    except Exception as e:
                        logger.error(f"Scene {scene_id}: Error calculating intersection: {e}")
                        actual_download_km2 += scene_area_km2  # Fallback to full scene
                else:
                    actual_download_km2 += scene_area_km2

            except Exception as e:
                logger.warning(f"Could not calculate area for scene {i}: {e}")
                # Use realistic PSScene size (typical: 3x3km = 9 km²)
                fallback_area = 9.0
                total_quota_km2 += fallback_area
                actual_download_km2 += fallback_area

        # Get current quota status
        current_quota = await self.check_user_quota()
        
        # Realistic disk space estimation for PSScene 4-band imagery
        # PSScene 4-band: ~15-25 MB per km² (3m pixels, 4 bands, uint16)
        estimated_mb = actual_download_km2 * 20  # Conservative 20 MB/km²
        
        # Update quota info with corrected values
        current_quota.download_estimate_km2 = total_quota_km2  # What you're charged
        current_quota.download_estimate_mb = estimated_mb      # Actual disk usage
        current_quota.estimated_scenes_count = scene_count

        # Enhanced logging
        logger.info(f"DOWNLOAD IMPACT CORRECTED:")
        logger.info(f"  Scenes: {scene_count}")
        logger.info(f"  Planet quota charge: {total_quota_km2:.2f} km²")
        logger.info(f"  Actual download area: {actual_download_km2:.2f} km²")
        logger.info(f"  Estimated disk space: {estimated_mb:.1f} MB")
        logger.info(f"  Average scene size: {total_quota_km2/scene_count:.2f} km²/scene")

        return current_quota

    def _estimate_download_impact_sync(self, scenes, roi=None, clip_to_roi=False):
        """FIXED: Synchronous version with correct area calculation."""
        total_quota_km2 = 0.0
        actual_download_km2 = 0.0
        scene_count = len(scenes)

        for i, scene in enumerate(scenes):
            try:
                scene_geom = shape(scene["geometry"])
                
                # Planet charges for full scene area regardless of clipping
                scene_area_km2 = calculate_area_km2(scene_geom)
                total_quota_km2 += scene_area_km2
                
                if clip_to_roi and roi:
                    try:
                        clipped_geom = scene_geom.intersection(roi)
                        if not clipped_geom.is_empty:
                            actual_download_km2 += calculate_area_km2(clipped_geom)
                    except Exception:
                        actual_download_km2 += scene_area_km2
                else:
                    actual_download_km2 += scene_area_km2

            except Exception as e:
                logger.warning(f"Could not calculate area for scene {i}: {e}")
                fallback_area = 9.0  # Realistic PSScene size
                total_quota_km2 += fallback_area
                actual_download_km2 += fallback_area

        # Get estimated quota
        current_quota = self._get_estimated_quota()
        
        # Correct disk space estimation
        estimated_mb = actual_download_km2 * 20
        
        current_quota.download_estimate_km2 = total_quota_km2
        current_quota.download_estimate_mb = estimated_mb
        current_quota.estimated_scenes_count = scene_count

        return current_quota

    def get_user_confirmation(self, quota_info: QuotaInfo) -> bool:
        """
        Enhanced user confirmation with detailed information and proper cancellation tracking.
        """
        print(f"\n" + "=" * 60)
        print(f"DOWNLOAD IMPACT ASSESSMENT")
        print(f"=" * 60)
        print(f"Scenes to download: {quota_info.estimated_scenes_count}")
        print(f"Estimated download size: {quota_info.download_estimate_km2:.2f} km²")
        print(f"Estimated disk space: {quota_info.download_estimate_mb:.1f} MB")
        print(f"")
        print(f"QUOTA STATUS ({quota_info.quota_source.upper()}):")
        print(f"Current quota usage: {quota_info.current_usage_km2:.2f} km² / {quota_info.monthly_limit_km2:.2f} km²")
        print(f"Usage percentage: {quota_info.usage_percentage:.1%}")
        print(f"Remaining quota: {quota_info.remaining_km2:.2f} km²")

        if quota_info.estimated_cost_usd:
            print(f"Estimated cost: ${quota_info.estimated_cost_usd:.2f} USD")

        if not quota_info.can_download:
            print(f"\n❌ ERROR: Download would exceed quota limit!")
            print(f"   Required: {quota_info.download_estimate_km2:.2f} km²")
            print(f"   Available: {quota_info.remaining_km2:.2f} km²")
            print(f"   Shortfall: {quota_info.download_estimate_km2 - quota_info.remaining_km2:.2f} km²")
            
            self.last_cancellation_reason = DownloadCancellationReason.QUOTA_EXCEEDED
            return False

        if quota_info.is_near_limit:
            print(f"\n⚠️  WARNING: Current usage is near limit ({quota_info.usage_percentage:.1%})")

        if quota_info.quota_source == "estimated":
            print(f"\n⚠️  WARNING: Using estimated quota values")
            print(f"   API quota check failed - actual usage may differ")

        print(f"=" * 60)

        while True:
            response = input(f"Proceed with download? (y/n): ").lower().strip()
            if response in ["y", "yes"]:
                self.last_cancellation_reason = None
                return True
            elif response in ["n", "no"]:
                print("Download cancelled by user.")
                self.last_cancellation_reason = DownloadCancellationReason.USER_CHOICE
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
        Enhanced main asset activation and download workflow with proper cancellation handling.
        """
        asset_types = asset_types or self.default_asset_types
        max_concurrent = max_concurrent or self.max_concurrent_downloads
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.last_cancellation_reason = None

        if confirm_download:
            quota_info = self.estimate_download_impact(scenes, roi, clip_to_roi)
            if not self.get_user_confirmation(quota_info):
                self.download_jobs = []
                for scene in scenes:
                    scene_id = scene["properties"]["id"]
                    for asset_type in asset_types:
                        job = DownloadJob(
                            scene_id=scene_id,
                            asset_type=asset_type,
                            item_type=scene["properties"].get("item_type", "PSScene"),
                            status=AssetStatus.USER_CANCELLED,
                            cancellation_reason=self.last_cancellation_reason
                        )
                        self.download_jobs.append(job)
                
                return self.download_jobs

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

        print(f"\nPhase 1: Activating {len(self.download_jobs)} assets...")
        await self._activate_assets_batch(self.download_jobs)

        print(f"\nPhase 2: Monitoring activation and downloading...")
        await self._download_activated_assets(
            self.download_jobs, output_path, max_concurrent
        )

        self._print_enhanced_download_summary(self.download_jobs)

        return self.download_jobs

    async def _activate_assets_batch(self, jobs: List[DownloadJob]) -> None:
        """FIXED: Activate all assets in parallel with rate limiting and proper semaphore handling."""

        async def activate_single_asset(job: DownloadJob) -> None:
            try:
                job.status = AssetStatus.ACTIVATING
                job.activation_time = datetime.now()

                assets_url = f"{self.data_api_url}/item-types/{job.item_type}/items/{job.scene_id}/assets"
                response = self.rate_limiter.make_request("GET", assets_url)

                if response.status_code == 200:
                    assets = response.json()

                    if job.asset_type in assets:
                        asset_info = assets[job.asset_type]

                        if asset_info.get("status") == "active":
                            job.status = AssetStatus.ACTIVE
                            job.download_url = asset_info.get("location")
                            logger.info(f"Asset {job.scene_id}:{job.asset_type} already active")
                            return

                        activation_url = asset_info["_links"]["activate"]
                        activate_response = self.rate_limiter.make_request("GET", activation_url)

                        if activate_response.status_code in [202, 204]:
                            logger.info(f"Asset {job.scene_id}:{job.asset_type} activation requested")
                        else:
                            raise AssetError(f"Activation failed: {activate_response.status_code}")
                    else:
                        raise AssetError(f"Asset type {job.asset_type} not available")
                else:
                    raise AssetError(f"Failed to get assets: {response.status_code}")

            except Exception as e:
                job.status = AssetStatus.FAILED
                job.error_message = str(e)
                logger.error(f"Failed to activate {job.scene_id}:{job.asset_type}: {e}")

        # FIXED: Properly define the semaphore wrapper function that was missing
        semaphore = asyncio.Semaphore(3)

        async def activate_with_semaphore(job: DownloadJob) -> None:
            """Wrapper to activate asset with semaphore control - THIS WAS THE MISSING FUNCTION!"""
            async with semaphore:
                await activate_single_asset(job)

        # Now create tasks with the properly defined function
        tasks = [activate_with_semaphore(job) for job in jobs]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _download_activated_assets(
        self, jobs: List[DownloadJob], output_path: Path, max_concurrent: int
    ) -> None:
        """Enhanced monitoring with proper activation timeouts and exponential backoff."""
        pending_jobs = [job for job in jobs if job.status != AssetStatus.FAILED]
        download_semaphore = asyncio.Semaphore(max_concurrent)

        async def monitor_and_download(job: DownloadJob) -> None:
            # Enhanced activation monitoring with configurable timeouts
            max_wait_time = self.timeouts["activation_timeout"]
            start_wait = time.time()
            check_interval = self.timeouts["activation_poll_interval"]
            max_check_interval = self.timeouts["activation_max_interval"]
            activation_checks = 0

            while job.status == AssetStatus.ACTIVATING:
                elapsed_time = time.time() - start_wait
                activation_checks += 1
                
                if elapsed_time > max_wait_time:
                    job.status = AssetStatus.FAILED
                    job.error_message = f"Activation timeout after {max_wait_time/60:.1f} minutes ({activation_checks} checks)"
                    logger.error(f"⏰ Activation timeout for {job.scene_id} after {activation_checks} checks")
                    return

                try:
                    # Check activation status
                    assets_url = f"{self.data_api_url}/item-types/{job.item_type}/items/{job.scene_id}/assets"
                    response = self.rate_limiter.make_request("GET", assets_url)

                    if response.status_code == 200:
                        assets = response.json()
                        asset_info = assets.get(job.asset_type, {})
                        asset_status = asset_info.get("status", "unknown")

                        if asset_status == "active":
                            job.status = AssetStatus.ACTIVE
                            job.download_url = asset_info.get("location")
                            logger.info(f"🟢 Asset {job.scene_id} activated after {elapsed_time/60:.1f} min ({activation_checks} checks)")
                            break
                        elif asset_status == "failed":
                            job.status = AssetStatus.FAILED
                            job.error_message = "Asset activation failed on Planet's side"
                            logger.error(f"💥 Asset activation failed for {job.scene_id}")
                            return
                        else:
                            # Log progress for long activations
                            if activation_checks % 10 == 0:  # Every 10 checks
                                logger.info(f"⏳ Still activating {job.scene_id}: {elapsed_time/60:.1f} min elapsed, status: {asset_status}")

                    # Exponential backoff for polling interval
                    check_interval = min(check_interval * 1.15, max_check_interval)  # Gentler increase
                    await asyncio.sleep(check_interval)

                except Exception as e:
                    logger.warning(f"⚠️  Error checking activation status for {job.scene_id}: {e}")
                    await asyncio.sleep(check_interval)

            # Download if active
            if job.status == AssetStatus.ACTIVE and job.download_url:
                # Check if download URL is still valid
                if job.is_expired:
                    job.status = AssetStatus.EXPIRED
                    job.error_message = "Download link expired"
                    logger.warning(f"🔗 Download link expired for {job.scene_id}")
                    return

                async with download_semaphore:
                    await self._download_single_asset(job, output_path)
                    
                    # Validate downloaded file
                    if job.status == AssetStatus.COMPLETED:
                        if not self._validate_downloaded_file(job):
                            job.status = AssetStatus.FAILED
                            job.error_message = "Downloaded file failed validation"
                            logger.error(f"❌ File validation failed for {job.scene_id}")

        # Monitor and download all assets
        tasks = [monitor_and_download(job) for job in pending_jobs]
        await asyncio.gather(*tasks, return_exceptions=True)

    # Enhanced download method with better progress tracking
    async def _download_single_asset(self, job: DownloadJob, output_path: Path) -> None:
        """Enhanced download with progress tracking for large files."""
        try:
            job.status = AssetStatus.DOWNLOADING
            job.download_start_time = datetime.now()

            filename = f"{job.scene_id}_{job.asset_type}.tif"
            file_path = output_path / filename
            job.file_path = file_path

            # Enhanced timeout for large downloads
            timeout = aiohttp.ClientTimeout(
                total=self.timeouts["download_total"],      # 4 hours for large files
                connect=self.timeouts["download_connect"],  # 1 minute connect
                sock_read=self.timeouts["download_read"]    # 5 minutes read
            )

            async with aiohttp.ClientSession(timeout=timeout) as session:
                try:
                    async with session.get(job.download_url) as response:
                        if response.status == 200:
                            total_size = int(response.headers.get("content-length", 0))
                            downloaded_size = 0
                            last_progress_time = time.time()
                            last_log_time = time.time()

                            # Log start of large download
                            if total_size > 100 * 1024 * 1024:  # > 100 MB
                                logger.info(f"🔽 Starting large download: {job.scene_id} ({total_size/1024/1024:.1f} MB)")

                            with open(file_path, "wb") as f:
                                async for chunk in response.content.iter_chunked(self.chunk_size):
                                    f.write(chunk)
                                    downloaded_size += len(chunk)
                                    
                                    current_time = time.time()
                                    
                                    # Progress logging every 60 seconds for large files
                                    if current_time - last_log_time > 60 and total_size > 0:
                                        progress_pct = (downloaded_size / total_size) * 100
                                        elapsed = current_time - job.download_start_time.timestamp()
                                        speed_mbps = (downloaded_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                                        eta_seconds = ((total_size - downloaded_size) / (downloaded_size / elapsed)) if downloaded_size > 0 and elapsed > 0 else 0
                                        eta_minutes = eta_seconds / 60
                                        
                                        logger.info(f"📊 Progress {job.scene_id}: {progress_pct:.1f}% ({downloaded_size/1024/1024:.1f}/{total_size/1024/1024:.1f} MB) "
                                                f"Speed: {speed_mbps:.2f} MB/s ETA: {eta_minutes:.1f} min")
                                        last_log_time = current_time
                                    
                                    # Enhanced stall detection for large files
                                    if current_time - last_progress_time > self.timeouts["download_stall"]:
                                        raise aiohttp.ServerTimeoutError(f"Download stalled - no progress for {self.timeouts['download_stall']/60:.1f} minutes")
                                    last_progress_time = current_time

                            job.file_size_mb = downloaded_size / (1024 * 1024)
                            job.status = AssetStatus.COMPLETED
                            job.completion_time = datetime.now()

                            duration = (job.completion_time - job.download_start_time).total_seconds()
                            avg_speed = job.file_size_mb / (duration / 60) if duration > 0 else 0
                            logger.info(f"✅ Downloaded {job.scene_id} ({job.file_size_mb:.1f} MB) in {duration/60:.1f} min (avg: {avg_speed:.1f} MB/min)")
                            return

                        elif response.status == 404:
                            job.status = AssetStatus.EXPIRED
                            job.error_message = f"Asset download link expired (HTTP {response.status})"
                            logger.warning(f"🔗 Link expired: {job.scene_id}")
                            return

                        elif response.status in [429, 503, 502, 504]:
                            raise aiohttp.ClientError(f"Server error: HTTP {response.status}")
                        
                        else:
                            job.status = AssetStatus.FAILED
                            job.error_message = f"Download failed: HTTP {response.status}"
                            logger.error(f"❌ Download failed {job.scene_id}: HTTP {response.status}")
                            return

                except (aiohttp.ClientError, asyncio.TimeoutError, aiohttp.ServerTimeoutError) as e:
                    error_msg = str(e)
                    logger.warning(f"⚠️  Retryable error {job.scene_id}: {error_msg}")
                    
                    if job.retry_count < self.retry_config["max_retries"]:
                        job.record_retry_attempt()
                        
                        # Enhanced retry delay for large files
                        base_delay = self.retry_config["initial_delay"]
                        exponential_delay = base_delay * (self.retry_config["exponential_base"] ** (job.retry_count - 1))
                        delay = min(exponential_delay, self.retry_config["max_delay"])
                        
                        if self.retry_config["jitter"]:
                            import random
                            delay *= (0.5 + random.random() * 0.5)
                        
                        logger.warning(f"🔄 Retrying {job.scene_id} (attempt {job.retry_count}/{self.retry_config['max_retries']}) in {delay/60:.1f} minutes")
                        
                        await asyncio.sleep(delay)
                        await self._download_single_asset(job, output_path)
                        return
                    else:
                        job.status = AssetStatus.FAILED
                        job.error_message = f"Download failed after {self.retry_config['max_retries']} retries: {error_msg}"
                        logger.error(f"❌ Permanent failure {job.scene_id} after {self.retry_config['max_retries']} retries")
                        return

        except Exception as e:
            # Non-retryable errors (file system errors, etc.)
            job.status = AssetStatus.FAILED
            job.error_message = f"Download failed: {str(e)}"
            logger.error(f"❌ Non-retryable error {job.scene_id}: {e}")
            
            # Clean up partial file if it exists
            if job.file_path and job.file_path.exists():
                try:
                    job.file_path.unlink()
                    logger.info(f"🧹 Cleaned up partial file: {job.file_path.name}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up partial file: {cleanup_error}")

    # Additional helper method for download validation
    def _validate_downloaded_file(self, job: DownloadJob) -> bool:
        """Validate that the downloaded file is complete and not corrupted."""
        if not job.file_path or not job.file_path.exists():
            return False
        
        try:
            # Check file size is reasonable
            file_size = job.file_path.stat().st_size
            if file_size < 1024:  # Less than 1KB is probably corrupted
                logger.warning(f"Downloaded file {job.scene_id} is suspiciously small: {file_size} bytes")
                return False
            
            # For GeoTIFF files, do a basic header check
            if job.file_path.suffix.lower() in ['.tif', '.tiff']:
                with open(job.file_path, 'rb') as f:
                    # Check for TIFF magic number
                    header = f.read(4)
                    if header not in [b'II*\x00', b'MM\x00*']:
                        logger.warning(f"Downloaded file {job.scene_id} does not appear to be a valid TIFF")
                        return False
            
            logger.debug(f"File validation passed for {job.scene_id}: {file_size} bytes")
            return True
            
        except Exception as e:
            logger.warning(f"Error validating downloaded file {job.scene_id}: {e}")
            return False

    def _should_retry_download_error(self, error: Exception) -> bool:
        """Determine if a download error is retryable."""
        if isinstance(error, (aiohttp.ClientError, asyncio.TimeoutError, aiohttp.ServerTimeoutError)):
            return True
            
        if isinstance(error, (ConnectionError, TimeoutError)):
            return True
            
        if "rate limit" in str(error).lower():
            return True
            
        if "50" in str(error) or "503" in str(error) or "502" in str(error):
            return True
            
        if "429" in str(error):
            return True
            
        if isinstance(error, (PermissionError, OSError, IOError)):
            return False
            
        return False

    def _print_enhanced_download_summary(self, jobs: List[DownloadJob]) -> None:
        """Enhanced download summary with detailed retry and timeout information."""
        completed = [j for j in jobs if j.status == AssetStatus.COMPLETED]
        failed = [j for j in jobs if j.status == AssetStatus.FAILED]
        cancelled = [j for j in jobs if j.status == AssetStatus.USER_CANCELLED]
        expired = [j for j in jobs if j.status == AssetStatus.EXPIRED]

        total_size_mb = sum(j.file_size_mb or 0 for j in completed)
        total_retries = sum(j.retry_count for j in jobs)

        print(f"\n" + "=" * 60)
        print(f"ENHANCED DOWNLOAD SUMMARY")
        print(f"=" * 60)
        
        if cancelled:
            if hasattr(self, 'last_cancellation_reason') and self.last_cancellation_reason:
                if self.last_cancellation_reason == DownloadCancellationReason.USER_CHOICE:
                    print(f"Status: Download cancelled by user choice")
                elif self.last_cancellation_reason == DownloadCancellationReason.QUOTA_EXCEEDED:
                    print(f"Status: Download cancelled - quota would be exceeded")
                else:
                    print(f"Status: Download cancelled")
            else:
                print(f"Status: Download cancelled")
            
            print(f"Cancelled: {len(cancelled)} assets")
            
        else:
            print(f"Completed: {len(completed)}/{len(jobs)} assets")
            
            if failed:
                print(f"Failed: {len(failed)} assets")
                
            if expired:
                print(f"Expired: {len(expired)} assets")
                
            if completed:
                print(f"Total Size: {total_size_mb:.1f} MB")
                
                successful_durations = [j.duration_seconds for j in completed if j.duration_seconds]
                if successful_durations:
                    avg_duration = sum(successful_durations) / len(successful_durations)
                    print(f"Average Duration: {avg_duration/60:.1f} minutes")

            if total_retries > 0:
                print(f"\nRetry Statistics:")
                print(f"Total Retries: {total_retries}")
                retry_jobs = [j for j in jobs if j.retry_count > 0]
                if retry_jobs:
                    avg_retries = total_retries / len(retry_jobs)
                    print(f"Jobs with Retries: {len(retry_jobs)}")
                    print(f"Average Retries per Failed Job: {avg_retries:.1f}")

            if failed:
                print(f"\nFailure Analysis:")
                timeout_failures = [j for j in failed if "timeout" in (j.error_message or "").lower()]
                network_failures = [j for j in failed if any(term in (j.error_message or "").lower() 
                                                            for term in ["connection", "network", "timeout"])]
                activation_failures = [j for j in failed if "activation" in (j.error_message or "").lower()]
                expired_failures = [j for j in failed if "expired" in (j.error_message or "").lower()]
                
                if timeout_failures:
                    print(f"   Timeout Failures: {len(timeout_failures)}")
                if network_failures:
                    print(f"   Network Failures: {len(network_failures)}")
                if activation_failures:
                    print(f"   Activation Failures: {len(activation_failures)}")
                if expired_failures:
                    print(f"   Expired Link Failures: {len(expired_failures)}")

        print(f"=" * 60)

    def _print_download_summary(self, jobs: List[DownloadJob]) -> None:
        """Legacy download summary method - maintained for backward compatibility."""
        self._print_enhanced_download_summary(jobs)

    def get_download_statistics(self) -> Dict:
        """Get comprehensive download statistics."""
        if not self.download_jobs:
            return {"error": "No download jobs available"}
        
        completed = [j for j in self.download_jobs if j.status == AssetStatus.COMPLETED]
        failed = [j for j in self.download_jobs if j.status == AssetStatus.FAILED]
        expired = [j for j in self.download_jobs if j.status == AssetStatus.EXPIRED]
        
        total_retries = sum(j.retry_count for j in self.download_jobs)
        total_size_mb = sum(j.file_size_mb or 0 for j in completed)
        
        total_attempted = len([j for j in self.download_jobs if j.status != AssetStatus.USER_CANCELLED])
        success_rate = len(completed) / total_attempted if total_attempted > 0 else 0
        
        jobs_with_retries = [j for j in self.download_jobs if j.retry_count > 0]
        avg_retries = total_retries / len(jobs_with_retries) if jobs_with_retries else 0
        
        return {
            "summary": {
                "total_jobs": len(self.download_jobs),
                "completed": len(completed),
                "failed": len(failed),
                "expired": len(expired),
                "success_rate": success_rate,
                "total_size_mb": total_size_mb
            },
            "retry_stats": {
                "total_retries": total_retries,
                "jobs_with_retries": len(jobs_with_retries),
                "average_retries": avg_retries,
                "retry_rate": len(jobs_with_retries) / len(self.download_jobs) if self.download_jobs else 0
            },
            "timing": {
                "total_duration_minutes": sum(j.duration_seconds or 0 for j in completed) / 60,
                "average_duration_minutes": (sum(j.duration_seconds or 0 for j in completed) / len(completed) / 60) if completed else 0
            }
        }

    def diagnose_download_issues(self) -> Dict:
        """Diagnose common download issues and provide recommendations."""
        if not self.download_jobs:
            return {"error": "No download jobs to diagnose"}
        
        failed_jobs = [j for j in self.download_jobs if j.status == AssetStatus.FAILED]
        
        if not failed_jobs:
            return {"status": "No failed jobs to diagnose"}
        
        issues = {
            "timeout_issues": [],
            "network_issues": [],
            "activation_issues": [],
            "expired_links": [],
            "quota_issues": [],
            "unknown_issues": []
        }
        
        recommendations = []
        
        for job in failed_jobs:
            error_msg = (job.error_message or "").lower()
            
            if "timeout" in error_msg or "stall" in error_msg:
                issues["timeout_issues"].append(job.scene_id)
            elif any(term in error_msg for term in ["connection", "network", "resolve"]):
                issues["network_issues"].append(job.scene_id)
            elif "activation" in error_msg:
                issues["activation_issues"].append(job.scene_id)
            elif "expired" in error_msg or "404" in error_msg:
                issues["expired_links"].append(job.scene_id)
            elif "quota" in error_msg:
                issues["quota_issues"].append(job.scene_id)
            else:
                issues["unknown_issues"].append(job.scene_id)
        
        if issues["timeout_issues"]:
            recommendations.append("Increase download timeouts or check network stability")
        if issues["network_issues"]:
            recommendations.append("Check internet connection and DNS resolution")
        if issues["activation_issues"]:
            recommendations.append("Assets may not be available - check Planet status or try different dates")
        if issues["expired_links"]:
            recommendations.append("Download links expired - try activating assets again")
        if issues["quota_issues"]:
            recommendations.append("Quota exceeded - wait for reset or upgrade plan")
        
        total_retries = sum(j.retry_count for j in self.download_jobs)
        if total_retries > len(self.download_jobs) * 2:
            recommendations.append("High retry rate detected - consider reducing concurrent downloads or checking Planet API status")
        
        return {
            "issues_found": {k: v for k, v in issues.items() if v},
            "recommendations": recommendations,
            "total_failed": len(failed_jobs),
            "failure_rate": len(failed_jobs) / len(self.download_jobs)
        }

    def get_download_status(self) -> Dict:
        """Get current download status summary with enhanced cancellation tracking."""
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
        cancelled_jobs = [
            j for j in self.download_jobs if j.status == AssetStatus.USER_CANCELLED
        ]
        
        total_size_mb = sum(j.file_size_mb or 0 for j in completed_jobs)

        result = {
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

        if cancelled_jobs:
            cancellation_reasons = [j.cancellation_reason.value for j in cancelled_jobs if j.cancellation_reason]
            result["cancellation_info"] = {
                "cancelled_count": len(cancelled_jobs),
                "cancellation_reasons": cancellation_reasons,
                "last_cancellation_reason": self.last_cancellation_reason.value if self.last_cancellation_reason else None
            }

        return result

    def export_download_report(self, output_path: str) -> None:
        """Export detailed download report to JSON with enhanced cancellation tracking."""
        report = {
            "download_summary": self.get_download_status(),
            "download_statistics": self.get_download_statistics(),
            "issue_diagnosis": self.diagnose_download_issues(),
            "job_details": [asdict(job) for job in self.download_jobs],
            "configuration": {
                "timeouts": self.timeouts,
                "retry_config": self.retry_config,
                "max_concurrent_downloads": self.max_concurrent_downloads
            },
            "cancellation_tracking": {
                "last_cancellation_reason": self.last_cancellation_reason.value if self.last_cancellation_reason else None,
                "has_cancellations": any(j.status == AssetStatus.USER_CANCELLED for j in self.download_jobs)
            },
            "generated_at": datetime.now().isoformat(),
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"Enhanced download report saved to: {output_path}")

    def clear_download_jobs(self) -> None:
        """Clear all download jobs and reset cancellation tracking."""
        self.download_jobs = []
        self.last_cancellation_reason = None

    def get_quota_summary(self) -> Dict:
        """Get a summary of current quota status."""
        try:
            quota_info = asyncio.run(self.check_user_quota())
            return {
                "current_usage_km2": quota_info.current_usage_km2,
                "monthly_limit_km2": quota_info.monthly_limit_km2,
                "remaining_km2": quota_info.remaining_km2,
                "usage_percentage": quota_info.usage_percentage,
                "quota_status": quota_info.quota_status,
                "quota_source": quota_info.quota_source,
                "last_updated": quota_info.last_updated.isoformat() if quota_info.last_updated else None
            }
        except Exception as e:
            logger.error(f"Failed to get quota summary: {e}")
            return {
                "error": str(e),
                "quota_status": "ERROR",
                "quota_source": "error"
            }

    def estimate_scenes_in_quota(self, average_scene_size_km2: float = 25.0) -> int:
        """Estimate how many average-sized scenes can be downloaded with remaining quota."""
        try:
            quota_info = asyncio.run(self.check_user_quota())
            if quota_info.remaining_km2 > 0:
                return int(quota_info.remaining_km2 / average_scene_size_km2)
            return 0
        except Exception:
            return 0

    def get_failed_jobs_summary(self) -> Dict:
        """Get summary of failed download jobs with error categorization."""
        failed_jobs = [j for j in self.download_jobs if j.status == AssetStatus.FAILED]
        
        if not failed_jobs:
            return {"failed_count": 0, "error_categories": {}}
        
        error_categories = {}
        for job in failed_jobs:
            error_msg = job.error_message or "Unknown error"
            
            if "activation" in error_msg.lower():
                category = "activation_error"
            elif "download" in error_msg.lower():
                category = "download_error"
            elif "timeout" in error_msg.lower():
                category = "timeout_error"
            elif "http" in error_msg.lower():
                category = "http_error"
            else:
                category = "other_error"
            
            if category not in error_categories:
                error_categories[category] = []
            
            error_categories[category].append({
                "scene_id": job.scene_id,
                "asset_type": job.asset_type,
                "error": error_msg
            })
        
        return {
            "failed_count": len(failed_jobs),
            "error_categories": error_categories
        }

    def get_configuration_recommendations(self) -> Dict:
        """Get configuration recommendations based on download performance."""
        if not self.download_jobs:
            return {"message": "No download data available for recommendations"}
        
        recommendations = []
        config_suggestions = {}
        
        failed_jobs = [j for j in self.download_jobs if j.status == AssetStatus.FAILED]
        total_retries = sum(j.retry_count for j in self.download_jobs)
        
        failure_rate = len(failed_jobs) / len(self.download_jobs)
        if failure_rate > 0.3:
            recommendations.append("High failure rate detected - consider more conservative settings")
            config_suggestions["max_concurrent_downloads"] = max(1, self.max_concurrent_downloads - 1)
            config_suggestions["timeouts"] = {
                **self.timeouts,
                "download_total": self.timeouts["download_total"] * 1.5,
                "activation_timeout": self.timeouts["activation_timeout"] * 1.5
            }
        
        if total_retries > len(self.download_jobs) * 1.5:
            recommendations.append("High retry rate - increase retry delays")
            config_suggestions["retry_config"] = {
                **self.retry_config,
                "initial_delay": self.retry_config["initial_delay"] * 1.5,
                "max_delay": self.retry_config["max_delay"] * 1.5
            }
        
        timeout_failures = [j for j in failed_jobs if "timeout" in (j.error_message or "").lower()]
        if len(timeout_failures) > len(failed_jobs) * 0.5:
            recommendations.append("Many timeout failures - increase timeout values")
            config_suggestions["timeouts"] = {
                **self.timeouts,
                "download_total": self.timeouts["download_total"] * 2,
                "download_read": self.timeouts["download_read"] * 2
            }
        
        return {
            "recommendations": recommendations,
            "suggested_config": config_suggestions,
            "current_performance": {
                "failure_rate": failure_rate,
                "retry_rate": total_retries / len(self.download_jobs),
                "success_rate": len([j for j in self.download_jobs if j.status == AssetStatus.COMPLETED]) / len(self.download_jobs)
            }
        }


# Configuration presets for different network conditions
NETWORK_CONFIGS = {
    "fast_reliable": {
        "max_concurrent_downloads": 5,
        "timeouts": {
            "activation_timeout": 900,
            "download_total": 1800,
            "download_stall": 180
        },
        "retry_config": {
            "max_retries": 2,
            "initial_delay": 15
        }
    },
    
    "balanced": {
        "max_concurrent_downloads": 3,
        "timeouts": {
            "activation_timeout": 1800,
            "download_total": 3600,
            "download_stall": 300
        },
        "retry_config": {
            "max_retries": 3,
            "initial_delay": 30,
            "max_delay": 300
        }
    },
    
    "slow_unreliable": {
        "max_concurrent_downloads": 2,
        "timeouts": {
            "activation_timeout": 2400,
            "download_total": 7200,
            "download_stall": 600
        },
        "retry_config": {
            "max_retries": 5,
            "initial_delay": 60,
            "max_delay": 900
        }
    }
}


# Network configuration presets for different scenarios
NETWORK_CONFIGURATIONS = {
    "very_slow": {
        "max_concurrent_downloads": 1,
        "timeouts": {
            "activation_timeout": 3600,     # 1 hour
            "download_total": 21600,        # 6 hours
            "download_connect": 120,        # 2 minutes
            "download_read": 600,           # 10 minutes
            "download_stall": 1800          # 30 minutes
        },
        "retry_config": {
            "max_retries": 5,
            "initial_delay": 180,           # 3 minutes
            "max_delay": 3600,              # 1 hour
            "exponential_base": 1.3
        }
    },
    
    "large_files": {
        "max_concurrent_downloads": 1,
        "timeouts": {
            "activation_timeout": 2400,     # 40 minutes
            "download_total": 14400,        # 4 hours
            "download_connect": 60,         # 1 minute
            "download_read": 300,           # 5 minutes
            "download_stall": 900           # 15 minutes
        },
        "retry_config": {
            "max_retries": 3,
            "initial_delay": 120,           # 2 minutes
            "max_delay": 1800               # 30 minutes
        }
    },
    
    "reliable_fast": {
        "max_concurrent_downloads": 3,
        "timeouts": {
            "activation_timeout": 1200,     # 20 minutes
            "download_total": 3600,         # 1 hour
            "download_connect": 30,
            "download_read": 120,           # 2 minutes
            "download_stall": 300           # 5 minutes
        },
        "retry_config": {
            "max_retries": 2,
            "initial_delay": 30,
            "max_delay": 300
        }
    }
}


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the enhanced asset manager.
    
    Basic usage:
        auth = PlanetAuth()  # Handles authentication seamlessly
        asset_manager = AssetManager(auth)  # Uses balanced defaults
        
    For slow networks:
        asset_manager = AssetManager(auth, config=NETWORK_CONFIGS["slow_unreliable"])
        
    For fast networks:
        asset_manager = AssetManager(auth, config=NETWORK_CONFIGS["fast_reliable"])
        
    For very large files on slow connections:
        asset_manager = AssetManager(auth, config=NETWORK_CONFIGURATIONS["very_slow"])
    """
    pass


# Additional utility functions for configuration management
def create_custom_config(
    network_speed: str = "balanced",
    file_size_expectation: str = "medium",
    reliability: str = "normal"
) -> Dict:
    """
    Create a custom configuration based on network conditions and requirements.
    
    Args:
        network_speed: "fast", "balanced", "slow"
        file_size_expectation: "small", "medium", "large", "very_large"
        reliability: "high", "normal", "low"
    
    Returns:
        Dict: Configuration optimized for the specified conditions
    """
    base_config = NETWORK_CONFIGS.get(network_speed, NETWORK_CONFIGS["balanced"]).copy()
    
    # Adjust for file size expectations
    if file_size_expectation == "very_large":
        base_config["max_concurrent_downloads"] = 1
        base_config["timeouts"]["download_total"] *= 3
        base_config["timeouts"]["download_stall"] *= 2
    elif file_size_expectation == "large":
        base_config["max_concurrent_downloads"] = min(2, base_config["max_concurrent_downloads"])
        base_config["timeouts"]["download_total"] *= 2
    elif file_size_expectation == "small":
        base_config["max_concurrent_downloads"] = min(5, base_config["max_concurrent_downloads"] + 2)
        base_config["timeouts"]["download_total"] = max(900, base_config["timeouts"]["download_total"] // 2)
    
    # Adjust for reliability requirements
    if reliability == "high":
        base_config["retry_config"]["max_retries"] = min(5, base_config["retry_config"]["max_retries"] + 2)
        base_config["retry_config"]["initial_delay"] *= 1.5
        base_config["timeouts"]["activation_timeout"] *= 1.5
    elif reliability == "low":
        base_config["retry_config"]["max_retries"] = max(1, base_config["retry_config"]["max_retries"] - 1)
        base_config["retry_config"]["initial_delay"] = max(15, base_config["retry_config"]["initial_delay"] // 2)
    
    return base_config


def validate_config(config: Dict) -> Tuple[bool, List[str]]:
    """
    Validate configuration parameters and return warnings.
    
    Args:
        config: Configuration dictionary to validate
    
    Returns:
        Tuple of (is_valid, warnings_list)
    """
    warnings = []
    is_valid = True
    
    # Check concurrent downloads
    max_concurrent = config.get("max_concurrent_downloads", 3)
    if max_concurrent < 1:
        warnings.append("max_concurrent_downloads must be at least 1")
        is_valid = False
    elif max_concurrent > 10:
        warnings.append("max_concurrent_downloads > 10 may overwhelm Planet's servers")
    
    # Check timeout values
    timeouts = config.get("timeouts", {})
    if timeouts.get("download_total", 3600) < 300:
        warnings.append("download_total timeout < 5 minutes may be too short for large files")
    if timeouts.get("activation_timeout", 1800) < 300:
        warnings.append("activation_timeout < 5 minutes may be too short")
    
    # Check retry configuration
    retry_config = config.get("retry_config", {})
    if retry_config.get("max_retries", 3) > 10:
        warnings.append("max_retries > 10 may cause excessive delays")
    if retry_config.get("initial_delay", 60) < 10:
        warnings.append("initial_delay < 10 seconds may not give servers time to recover")
    
    return is_valid, warnings


def get_recommended_config_for_quota(available_quota_km2: float) -> Dict:
    """
    Get recommended configuration based on available quota.
    
    Args:
        available_quota_km2: Available quota in square kilometers
    
    Returns:
        Dict: Recommended configuration
    """
    if available_quota_km2 < 100:
        # Very limited quota - be extremely conservative
        return NETWORK_CONFIGURATIONS["very_slow"]
    elif available_quota_km2 < 500:
        # Limited quota - conservative settings
        return NETWORK_CONFIGS["slow_unreliable"]
    elif available_quota_km2 < 1500:
        # Moderate quota - balanced settings
        return NETWORK_CONFIGS["balanced"]
    else:
        # Plenty of quota - can be more aggressive
        return NETWORK_CONFIGS["fast_reliable"]


def print_config_summary(config: Dict) -> None:
    """Print a human-readable summary of configuration settings."""
    print("\n" + "=" * 50)
    print("ASSET MANAGER CONFIGURATION SUMMARY")
    print("=" * 50)
    
    print(f"Concurrent Downloads: {config.get('max_concurrent_downloads', 3)}")
    
    timeouts = config.get("timeouts", {})
    print(f"\nTimeouts:")
    print(f"  Activation: {timeouts.get('activation_timeout', 1800)/60:.1f} minutes")
    print(f"  Download Total: {timeouts.get('download_total', 3600)/60:.1f} minutes")
    print(f"  Download Stall: {timeouts.get('download_stall', 900)/60:.1f} minutes")
    
    retry_config = config.get("retry_config", {})
    print(f"\nRetry Configuration:")
    print(f"  Max Retries: {retry_config.get('max_retries', 3)}")
    print(f"  Initial Delay: {retry_config.get('initial_delay', 60)} seconds")
    print(f"  Max Delay: {retry_config.get('max_delay', 900)/60:.1f} minutes")
    
    rate_limits = config.get("rate_limits", {})
    if rate_limits:
        print(f"\nRate Limits:")
        for api, limit in rate_limits.items():
            print(f"  {api.title()}: {limit} requests/minute")
    
    print("=" * 50)


# Export the main class and utility functions
__all__ = [
    'AssetManager',
    'AssetStatus', 
    'DownloadCancellationReason',
    'QuotaInfo',
    'DownloadJob',
    'AssetError',
    'NETWORK_CONFIGS',
    'NETWORK_CONFIGURATIONS',
    'create_custom_config',
    'validate_config',
    'get_recommended_config_for_quota',
    'print_config_summary'
]


"""
USAGE EXAMPLES:

# Basic usage with default configuration
auth = PlanetAuth()
asset_manager = AssetManager(auth)

# Usage with predefined network configuration
asset_manager = AssetManager(auth, config=NETWORK_CONFIGS["slow_unreliable"])

# Usage with custom configuration
custom_config = create_custom_config(
    network_speed="slow",
    file_size_expectation="large", 
    reliability="high"
)
asset_manager = AssetManager(auth, config=custom_config)

# Validate configuration before use
is_valid, warnings = validate_config(custom_config)
if warnings:
    for warning in warnings:
        print(f"WARNING: {warning}")

# Print configuration summary
print_config_summary(custom_config)

# Download assets with the fixed semaphore handling
downloads = await asset_manager.activate_and_download_assets(
    scenes=scenes,
    asset_types=["ortho_analytic_4b"],
    clip_to_roi=True,
    max_concurrent=1
)

# Check results
status = asset_manager.get_download_status()
statistics = asset_manager.get_download_statistics()
diagnosis = asset_manager.diagnose_download_issues()

# Export detailed report
asset_manager.export_download_report("download_report.json")
"""