# Design Document: Community Issue Reporting Platform

## Overview

The Community Issue Reporting Platform is a full-stack web application that enables citizens to report civic issues through multiple input modalities (photo, voice, text) while leveraging AI to automatically classify, prioritize, and route issues to appropriate government departments. The system consists of three main interfaces: a responsive citizen web app, an authority dashboard for issue management, and a public transparency portal.

### Key Design Principles

1. **AI-First Approach**: Minimize manual intervention through intelligent classification and routing
2. **Real-Time Transparency**: Provide live updates to all stakeholders
3. **Mobile-First Design**: Optimize for mobile browsers as primary access method
4. **Scalable Architecture**: Support growing user base and issue volume
5. **Privacy by Design**: Protect citizen identity while maintaining accountability

## Architecture

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                             │
├──────────────────┬──────────────────┬──────────────────────────┤
│  Citizen Web App │ Authority Portal │ Public Dashboard         │
│  (React.js)      │ (React.js)       │ (React.js)              │
└────────┬─────────┴────────┬─────────┴──────────┬───────────────┘
         │                  │                     │
         └──────────────────┼─────────────────────┘
                            │
                    ┌───────▼────────┐
                    │  API Gateway   │
                    │  (Rate Limit,  │
                    │   Auth, CORS)  │
                    └───────┬────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
    ┌────▼─────┐    ┌──────▼──────┐    ┌─────▼──────┐
    │  Auth    │    │   Issue     │    │  Analytics │
    │  Service │    │   Service   │    │  Service   │
    └────┬─────┘    └──────┬──────┘    └─────┬──────┘
         │                 │                  │
         │          ┌──────▼──────┐           │
         │          │  AI Engine  │           │
         │          │  Service    │           │
         │          └──────┬──────┘           │
         │                 │                  │
         │          ┌──────▼──────┐           │
         │          │  Media      │           │
         │          │  Service    │           │
         │          └──────┬──────┘           │
         │                 │                  │
    ┌────▼─────────────────▼──────────────────▼────┐
    │           Database Layer                      │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
    │  │PostgreSQL│  │  Redis   │  │  Cloud   │   │
    │  │(Primary) │  │ (Cache)  │  │ Storage  │   │
    │  └──────────┘  └──────────┘  └──────────┘   │
    └───────────────────────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
    ┌────▼─────┐    ┌──────▼──────┐    ┌─────▼──────┐
    │  Email   │    │    SMS      │    │  Push      │
    │  Service │    │   Service   │    │  Notif.    │
    └──────────┘    └─────────────┘    └────────────┘
```

### Technology Stack

**Frontend:**
- React.js 18+ with TypeScript
- React Router for navigation
- React Query for data fetching and caching
- Leaflet.js or Google Maps API for mapping
- Tailwind CSS for styling
- PWA capabilities for offline support

**Backend:**
- FastAPI (Python 3.10+) for main API server
- Pydantic for data validation
- SQLAlchemy for ORM
- Alembic for database migrations
- Celery for async task processing
- Redis for caching and task queue

**AI/ML Stack:**
- PyTorch 2.0+ for model inference
- Transformers library for NLP models
- OpenCV for image preprocessing
- Whisper or DeepSpeech for speech-to-text
- ONNX Runtime for optimized inference

**Database:**
- PostgreSQL 15+ with PostGIS extension for geospatial queries
- Redis for session management and caching

**Infrastructure:**
- Docker for containerization
- Kubernetes for orchestration
- AWS S3 or Google Cloud Storage for media files
- CloudFront or Cloud CDN for content delivery
- AWS Lambda or Cloud Functions for serverless AI inference

**External Services:**
- Twilio for SMS notifications
- SendGrid for email notifications
- Google Maps API for geocoding and mapping
- Firebase Cloud Messaging for web push notifications

## Components and Interfaces

### 1. Authentication Service

**Responsibilities:**
- Generate and validate OTPs
- Manage user sessions
- Issue JWT tokens
- Handle rate limiting for authentication attempts

**Key Functions:**

```python
def send_otp(phone_number: str) -> OTPResponse:
    """
    Generate a 6-digit OTP, store it with 5-minute expiry in Redis,
    and send via SMS service.
    
    Returns: OTPResponse with request_id and expiry_time
    """

def verify_otp(phone_number: str, otp: str, request_id: str) -> AuthToken:
    """
    Verify OTP against stored value, check expiry and attempt count.
    On success, create user session and return JWT token.
    On failure, increment attempt count and raise appropriate error.
    
    Returns: AuthToken with JWT and refresh token
    """

def refresh_token(refresh_token: str) -> AuthToken:
    """
    Validate refresh token and issue new access token.
    
    Returns: New AuthToken
    """

def block_phone_number(phone_number: str, duration_minutes: int):
    """
    Temporarily block phone number after failed attempts.
    Store block in Redis with TTL.
    """
```

**API Endpoints:**
- `POST /api/auth/send-otp` - Request OTP
- `POST /api/auth/verify-otp` - Verify OTP and get token
- `POST /api/auth/refresh` - Refresh access token
- `POST /api/auth/logout` - Invalidate session

### 2. Issue Service

**Responsibilities:**
- Create and manage issues
- Handle status workflow transitions
- Coordinate with AI Engine for classification
- Manage issue assignments
- Track status history

**Key Functions:**

```python
def create_issue(
    user_id: str,
    description: Optional[str],
    location: GeoPoint,
    media_files: List[MediaFile]
) -> Issue:
    """
    Create new issue, trigger AI classification pipeline,
    detect duplicates, calculate initial priority score.
    
    Returns: Created Issue with assigned ID and initial status
    """

def update_issue_status(
    issue_id: str,
    new_status: IssueStatus,
    user_id: str,
    notes: Optional[str],
    evidence_photo: Optional[MediaFile]
) -> Issue:
    """
    Update issue status following workflow rules.
    Validate status transition is allowed.
    Record status history entry.
    Trigger notifications.
    
    Returns: Updated Issue
    """

def assign_issue(
    issue_id: str,
    department_id: str,
    assigned_by: str
) -> Issue:
    """
    Assign issue to department, update status to Assigned,
    send notification to department, check department capacity.
    
    Returns: Updated Issue
    """

def get_issues(
    filters: IssueFilters,
    pagination: Pagination
) -> PaginatedIssues:
    """
    Query issues with filters (category, severity, status, location, date range).
    Apply spatial queries for location-based filtering.
    
    Returns: Paginated list of issues
    """

def detect_duplicates(issue: Issue) -> List[Issue]:
    """
    Find similar issues within 500m radius, same category,
    submitted within 24 hours. Use geospatial query and
    text similarity comparison.
    
    Returns: List of potential duplicate issues
    """
```

**API Endpoints:**
- `POST /api/issues` - Create new issue
- `GET /api/issues` - List issues with filters
- `GET /api/issues/{id}` - Get issue details
- `PATCH /api/issues/{id}/status` - Update status
- `POST /api/issues/{id}/assign` - Assign to department
- `GET /api/issues/{id}/history` - Get status history
- `GET /api/users/me/issues` - Get user's issues

### 3. AI Engine Service

**Responsibilities:**
- Image classification
- Speech-to-text conversion
- NLP analysis and entity extraction
- Severity detection
- Priority score calculation
- Duplicate detection
- Spam filtering

**Key Functions:**

```python
def classify_image(image: ImageFile) -> Classification:
    """
    Run image through CNN/Vision Transformer model.
    Preprocess image (resize, normalize).
    Return category prediction with confidence score.
    
    Model: EfficientNet-B3 or Vision Transformer fine-tuned on
    civic issue dataset with 8 categories.
    
    Returns: Classification with category and confidence
    """

def transcribe_audio(audio: AudioFile) -> Transcription:
    """
    Convert audio to text using Whisper or DeepSpeech.
    Detect language and handle multi-lingual input.
    
    Returns: Transcription with text and language
    """

def analyze_text(text: str) -> TextAnalysis:
    """
    Use BERT-based classifier to extract:
    - Issue category
    - Severity indicators
    - Location entities
    - Urgency keywords
    
    Returns: TextAnalysis with extracted entities and classification
    """

def detect_severity(
    text: Optional[str],
    image_analysis: Optional[Classification],
    category: IssueCategory
) -> Severity:
    """
    Analyze text for urgency keywords and image for hazard detection.
    Apply rule-based logic combined with ML predictions.
    
    Priority rules:
    - Safety keywords (fire, accident, injury) → Critical
    - Infrastructure failure → High
    - Maintenance issues → Medium/Low
    - Image hazard detection overrides text
    
    Returns: Severity level (Low/Medium/High/Critical)
    """

def calculate_priority_score(
    severity: Severity,
    location: GeoPoint,
    duplicate_count: int,
    days_pending: int
) -> int:
    """
    Calculate priority score using weighted formula:
    Priority = severity_weight + location_risk + crowd_reports + time_penalty
    
    Weights:
    - Critical: 40, High: 30, Medium: 20, Low: 10
    - Location risk: 0-20 based on historical issue density
    - Crowd reports: 5 points per duplicate (max 25)
    - Time penalty: 1 point per day (max 15)
    
    Returns: Priority score (0-100)
    """

def detect_spam(
    text: Optional[str],
    image: Optional[ImageFile],
    user_history: UserHistory
) -> SpamScore:
    """
    Analyze for spam indicators:
    - Gibberish text detection
    - Unrelated image content
    - User submission rate
    - Historical trust score
    
    Returns: SpamScore with confidence and indicators
    """

def find_similar_issues(
    issue: Issue,
    radius_meters: int = 500,
    time_window_hours: int = 24
) -> List[SimilarIssue]:
    """
    Find similar issues using:
    - Geospatial query (PostGIS)
    - Category matching
    - Text similarity (cosine similarity on embeddings)
    - Time window filtering
    
    Returns: List of similar issues with similarity scores
    """
```

**AI Models:**

1. **Image Classification Model**
   - Architecture: EfficientNet-B3 or Vision Transformer (ViT-B/16)
   - Input: 224x224 RGB images
   - Output: 8 classes (Road, Water, Garbage, Electricity, Safety, Sanitation, Drainage, Streetlight)
   - Training: Transfer learning from ImageNet, fine-tuned on civic issue dataset
   - Deployment: ONNX format for fast inference

2. **Speech-to-Text Model**
   - Model: OpenAI Whisper (base or small model)
   - Languages: English, Hindi, and regional languages
   - Deployment: Run on GPU-enabled containers or serverless functions

3. **NLP Classification Model**
   - Architecture: BERT-base or DistilBERT
   - Tasks: Multi-label classification (category, severity, entities)
   - Training: Fine-tuned on labeled complaint text dataset
   - Deployment: ONNX format for CPU inference

4. **Spam Detection Model**
   - Architecture: Lightweight classifier (Random Forest or XGBoost)
   - Features: Text statistics, user behavior, image quality metrics
   - Training: Supervised learning on labeled spam/legitimate issues

**API Endpoints:**
- `POST /api/ai/classify-image` - Classify uploaded image
- `POST /api/ai/transcribe-audio` - Convert audio to text
- `POST /api/ai/analyze-text` - Extract entities and classify text
- `POST /api/ai/detect-duplicates` - Find similar issues

### 4. Media Service

**Responsibilities:**
- Upload and store media files
- Image compression and optimization
- Audio format conversion
- Generate signed URLs for secure access
- Manage media lifecycle (archival)

**Key Functions:**

```python
def upload_media(
    file: UploadFile,
    media_type: MediaType,
    user_id: str
) -> MediaFile:
    """
    Validate file type and size.
    Compress image or convert audio format.
    Upload to cloud storage (S3/GCS).
    Store metadata in database.
    
    Returns: MediaFile with storage URL and metadata
    """

def compress_image(image: ImageFile) -> ImageFile:
    """
    Resize to max 1920px width while maintaining aspect ratio.
    Compress to JPEG with 85% quality or WebP format.
    Reduce file size by 60-80% while maintaining readability.
    
    Returns: Compressed ImageFile
    """

def convert_audio(audio: AudioFile) -> AudioFile:
    """
    Convert to MP3 or Opus format with appropriate bitrate.
    Normalize audio levels.
    
    Returns: Converted AudioFile
    """

def generate_signed_url(media_id: str, expiry_minutes: int = 60) -> str:
    """
    Generate time-limited signed URL for secure media access.
    
    Returns: Signed URL string
    """

def archive_old_media(days_threshold: int = 90):
    """
    Move media files from resolved issues older than threshold
    to cold storage tier for cost optimization.
    """
```

**API Endpoints:**
- `POST /api/media/upload` - Upload media file
- `GET /api/media/{id}` - Get media file (returns signed URL)
- `DELETE /api/media/{id}` - Delete media file

### 5. Notification Service

**Responsibilities:**
- Send email notifications
- Send SMS notifications
- Send web push notifications
- Manage notification preferences
- Queue and batch notifications

**Key Functions:**

```python
def send_notification(
    user_id: str,
    notification_type: NotificationType,
    content: NotificationContent,
    channels: List[NotificationChannel]
):
    """
    Send notification through specified channels (email, SMS, push).
    Check user preferences for each channel.
    Queue notification for async processing.
    """

def send_email(to: str, subject: str, body: str, template: str):
    """
    Send email using SendGrid or similar service.
    Use HTML templates for formatted emails.
    """

def send_sms(phone: str, message: str):
    """
    Send SMS using Twilio or similar service.
    Keep message under 160 characters.
    """

def send_push_notification(user_id: str, title: str, body: str, data: dict):
    """
    Send web push notification using Firebase Cloud Messaging.
    Include action buttons for quick responses.
    """

def batch_notifications(notifications: List[Notification]):
    """
    Batch multiple notifications to same user to avoid spam.
    Send digest instead of individual notifications.
    """
```

**Notification Templates:**
- Issue submitted confirmation
- Status change updates
- Assignment notifications
- Resolution notifications
- Escalation alerts
- Emergency alerts

### 6. Analytics Service

**Responsibilities:**
- Generate reports and statistics
- Calculate performance metrics
- Create heatmaps
- Track resolution times
- Monitor AI model performance

**Key Functions:**

```python
def get_dashboard_stats(
    date_range: DateRange,
    filters: AnalyticsFilters
) -> DashboardStats:
    """
    Calculate aggregate statistics:
    - Total issues by status
    - Issues by category
    - Average resolution time
    - Resolution rate
    - Department performance
    
    Returns: DashboardStats object
    """

def generate_heatmap(
    date_range: DateRange,
    category: Optional[IssueCategory]
) -> HeatmapData:
    """
    Generate geographic heatmap data showing issue density.
    Use spatial clustering to identify hotspots.
    
    Returns: HeatmapData with coordinates and intensity values
    """

def calculate_area_scores(date_range: DateRange) -> List[AreaScore]:
    """
    Calculate cleanliness and safety scores for each ward/district.
    Score based on:
    - Issue density (lower is better)
    - Resolution speed (faster is better)
    - Issue severity distribution
    
    Returns: List of AreaScore objects
    """

def get_department_performance(
    department_id: str,
    date_range: DateRange
) -> DepartmentPerformance:
    """
    Calculate department-specific metrics:
    - Total issues handled
    - Average resolution time
    - Resolution rate
    - SLA compliance rate
    - Escalation count
    
    Returns: DepartmentPerformance object
    """

def track_ai_accuracy() -> AIMetrics:
    """
    Monitor AI model performance:
    - Classification accuracy by category
    - Confidence score distribution
    - Manual correction rate
    - False positive/negative rates
    
    Returns: AIMetrics object
    """
```

**API Endpoints:**
- `GET /api/analytics/dashboard` - Get dashboard statistics
- `GET /api/analytics/heatmap` - Get heatmap data
- `GET /api/analytics/area-scores` - Get area-wise scores
- `GET /api/analytics/department/{id}` - Get department performance
- `GET /api/analytics/ai-metrics` - Get AI model metrics
- `GET /api/analytics/export` - Export reports (CSV/PDF)

### 7. Escalation Service

**Responsibilities:**
- Monitor issue SLAs
- Trigger automatic escalations
- Send escalation notifications
- Track escalation history

**Key Functions:**

```python
def check_sla_violations():
    """
    Periodic job (runs every hour) to check for SLA violations:
    - Critical: 2 hours in Submitted
    - High: 24 hours in Assigned
    - Any: 7 days in In Progress
    
    Trigger escalations for violations.
    """

def escalate_issue(
    issue_id: str,
    escalation_level: EscalationLevel,
    reason: str
):
    """
    Escalate issue to higher authority level.
    Send notifications to escalation recipients.
    Record escalation in issue history.
    """

def get_escalation_recipients(
    issue: Issue,
    escalation_level: EscalationLevel
) -> List[User]:
    """
    Determine who should receive escalation notifications
    based on issue category and escalation level.
    
    Returns: List of users to notify
    """
```

## Data Models

### User Model

```python
class User:
    id: UUID
    phone_number: str  # Encrypted
    phone_country_code: str
    created_at: datetime
    last_login: datetime
    is_active: bool
    role: UserRole  # CITIZEN, AUTHORITY, ADMIN
    department_id: Optional[UUID]
    trust_score: float  # 0.0 to 1.0
    notification_preferences: NotificationPreferences
    
class NotificationPreferences:
    email_enabled: bool
    sms_enabled: bool
    push_enabled: bool
    email_address: Optional[str]
```

### Issue Model

```python
class Issue:
    id: UUID
    reporter_id: UUID
    description: Optional[str]
    location: GeoPoint  # PostGIS geography type
    location_address: str
    category: IssueCategory
    severity: Severity
    status: IssueStatus
    priority_score: int
    created_at: datetime
    updated_at: datetime
    verified_at: Optional[datetime]
    assigned_at: Optional[datetime]
    resolved_at: Optional[datetime]
    assigned_to_department: Optional[UUID]
    assigned_to_user: Optional[UUID]
    is_duplicate: bool
    parent_issue_id: Optional[UUID]  # If duplicate
    duplicate_count: int
    is_spam: bool
    spam_score: float
    is_emergency: bool
    escalation_level: int
    
class IssueCategory(Enum):
    ROAD = "road"
    WATER = "water"
    GARBAGE = "garbage"
    ELECTRICITY = "electricity"
    SAFETY = "safety"
    SANITATION = "sanitation"
    DRAINAGE = "drainage"
    STREETLIGHT = "streetlight"
    
class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    
class IssueStatus(Enum):
    SUBMITTED = "submitted"
    VERIFIED = "verified"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
```

### Media Model

```python
class MediaFile:
    id: UUID
    issue_id: UUID
    media_type: MediaType  # IMAGE, AUDIO
    file_name: str
    file_size: int
    mime_type: str
    storage_url: str
    storage_bucket: str
    storage_key: str
    uploaded_at: datetime
    is_archived: bool
    
class MediaType(Enum):
    IMAGE = "image"
    AUDIO = "audio"
```

### AI Classification Model

```python
class AIClassification:
    id: UUID
    issue_id: UUID
    media_id: Optional[UUID]
    classification_type: ClassificationType  # IMAGE, TEXT, AUDIO
    predicted_category: IssueCategory
    confidence_score: float
    predicted_severity: Severity
    severity_confidence: float
    model_version: str
    created_at: datetime
    was_corrected: bool
    corrected_category: Optional[IssueCategory]
    corrected_by: Optional[UUID]
    
class ClassificationType(Enum):
    IMAGE = "image"
    TEXT = "text"
    AUDIO = "audio"
```

### Status History Model

```python
class StatusHistory:
    id: UUID
    issue_id: UUID
    from_status: Optional[IssueStatus]
    to_status: IssueStatus
    changed_by: UUID
    changed_at: datetime
    notes: Optional[str]
    evidence_media_id: Optional[UUID]
```

### Department Model

```python
class Department:
    id: UUID
    name: str
    category: IssueCategory
    contact_email: str
    contact_phone: str
    max_capacity: int  # Max concurrent issues
    current_load: int
    is_active: bool
```

### Escalation Model

```python
class Escalation:
    id: UUID
    issue_id: UUID
    escalation_level: int
    escalated_at: datetime
    escalated_to: List[UUID]  # User IDs
    reason: str
    acknowledged_at: Optional[datetime]
    acknowledged_by: Optional[UUID]
```

### Database Schema

**Key Indexes:**
- `issues.location` - Spatial index (GiST) for geographic queries
- `issues.status, issues.category` - Composite index for filtering
- `issues.created_at` - Index for time-based queries
- `issues.priority_score` - Index for sorting by priority
- `users.phone_number` - Unique index for authentication
- `status_history.issue_id, status_history.changed_at` - Composite index for history queries

**Relationships:**
- User → Issues (one-to-many)
- Issue → MediaFiles (one-to-many)
- Issue → AIClassifications (one-to-many)
- Issue → StatusHistory (one-to-many)
- Issue → Issue (self-referential for duplicates)
- Department → Issues (one-to-many)
- Issue → Escalations (one-to-many)

## 
AI Workflow Pipeline

### Issue Submission Pipeline

```
User Submits Issue
       ↓
[Validate Input]
       ↓
[Upload Media to Storage] ← Parallel Processing
       ↓
[Image Classification] ← If image present
       ↓
[Audio Transcription] ← If audio present
       ↓
[Text Analysis] ← From transcription or direct text
       ↓
[Severity Detection] ← Combine image + text analysis
       ↓
[Duplicate Detection] ← Geospatial + text similarity
       ↓
[Spam Detection] ← Check user history + content
       ↓
[Calculate Priority Score]
       ↓
[Create Issue Record]
       ↓
[Send Confirmation Notification]
```

### Classification Pipeline Details

**Image Classification:**
1. Receive uploaded image
2. Preprocess: Resize to 224x224, normalize pixel values
3. Run through EfficientNet-B3 model
4. Get category predictions with confidence scores
5. If confidence < 0.6, flag for manual review
6. Store classification result

**Audio Processing:**
1. Receive audio file
2. Convert to required format (16kHz, mono)
3. Run through Whisper model
4. Get transcription with language detection
5. If transcription confidence low, flag for manual review
6. Pass transcription to text analysis

**Text Analysis:**
1. Receive text (from transcription or direct input)
2. Tokenize and preprocess
3. Run through BERT classifier
4. Extract: category, severity indicators, location entities
5. Combine with image classification if available
6. Return final classification

**Duplicate Detection:**
1. Query issues within 500m radius (PostGIS spatial query)
2. Filter by same category
3. Filter by time window (last 24 hours)
4. Calculate text similarity using embeddings
5. If similarity > 0.85, mark as duplicate
6. Link to parent issue and increment duplicate count

**Priority Calculation:**
1. Get severity weight (Critical=40, High=30, Medium=20, Low=10)
2. Query historical issue density for location (0-20 points)
3. Add duplicate count weight (5 points each, max 25)
4. Add time penalty (1 point per day, max 15)
5. Sum all weights to get final priority score (0-100)

## Error Handling

### Error Categories

1. **Client Errors (4xx)**
   - 400 Bad Request: Invalid input data
   - 401 Unauthorized: Missing or invalid authentication
   - 403 Forbidden: Insufficient permissions
   - 404 Not Found: Resource doesn't exist
   - 429 Too Many Requests: Rate limit exceeded

2. **Server Errors (5xx)**
   - 500 Internal Server Error: Unexpected server error
   - 502 Bad Gateway: Upstream service failure
   - 503 Service Unavailable: Service temporarily down
   - 504 Gateway Timeout: Request timeout

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "Image file size exceeds 10MB limit",
    "details": {
      "field": "image",
      "max_size": "10MB",
      "provided_size": "15MB"
    },
    "request_id": "req_abc123"
  }
}
```

### Error Handling Strategies

**Authentication Errors:**
- Invalid OTP: Return clear error message, show remaining attempts
- Expired OTP: Prompt user to request new OTP
- Blocked phone: Show block duration and reason
- Invalid token: Clear local storage and redirect to login

**Media Upload Errors:**
- File too large: Show size limit and suggest compression
- Invalid format: List supported formats
- Upload failure: Retry with exponential backoff (3 attempts)
- Storage service down: Queue upload for later retry

**AI Processing Errors:**
- Model inference failure: Flag issue for manual classification
- Low confidence: Mark for manual review but allow submission
- Timeout: Use fallback classification based on text keywords
- Model unavailable: Queue for processing when service recovers

**Database Errors:**
- Connection failure: Retry with exponential backoff
- Deadlock: Retry transaction
- Constraint violation: Return specific validation error
- Query timeout: Optimize query or return partial results

**External Service Errors:**
- SMS service down: Queue notification for retry
- Email service down: Queue notification for retry
- Maps API failure: Use cached geocoding or allow manual entry
- Storage service down: Queue media upload for retry

### Retry Policies

**Exponential Backoff:**
- Initial delay: 1 second
- Max delay: 32 seconds
- Max attempts: 5
- Jitter: Random 0-1 second added to prevent thundering herd

**Circuit Breaker:**
- Failure threshold: 5 consecutive failures
- Timeout: 30 seconds
- Half-open state: Allow 1 test request after timeout
- Success threshold: 2 consecutive successes to close circuit

### Logging and Monitoring

**Log Levels:**
- ERROR: System errors requiring immediate attention
- WARN: Degraded functionality or approaching limits
- INFO: Normal operations and state changes
- DEBUG: Detailed diagnostic information

**Key Metrics to Monitor:**
- API response times (p50, p95, p99)
- Error rates by endpoint
- AI model inference latency
- Database query performance
- Queue depths (notification, AI processing)
- Storage usage and costs
- Active user sessions

**Alerts:**
- Error rate > 5% for 5 minutes
- API response time p95 > 2 seconds
- Database connection pool exhausted
- AI model accuracy < 80%
- Storage usage > 80% capacity
- Queue depth > 1000 items

## Testing Strategy

The testing strategy employs a dual approach combining unit tests for specific examples and edge cases with property-based tests for universal correctness properties. This ensures both concrete bug detection and general correctness verification.

### Testing Framework

**Unit Testing:**
- Frontend: Jest + React Testing Library
- Backend: pytest with pytest-asyncio
- Coverage target: 80% code coverage

**Property-Based Testing:**
- Backend: Hypothesis (Python)
- Frontend: fast-check (TypeScript)
- Configuration: Minimum 100 iterations per property test

**Integration Testing:**
- API testing: pytest with httpx
- End-to-end: Playwright or Cypress

**Load Testing:**
- Tool: Locust or k6
- Scenarios: Normal load, peak load, stress test

### Test Organization

**Unit Tests:**
- Test specific examples and edge cases
- Focus on error conditions and boundary values
- Test integration points between components
- Keep tests fast and isolated

**Property-Based Tests:**
- Test universal properties across all inputs
- Use randomized input generation
- Verify invariants and correctness properties
- Each test references design document property
- Tag format: `# Feature: community-issue-reporting-platform, Property {N}: {property_text}`

### Key Test Scenarios

**Authentication:**
- Valid OTP flow
- Expired OTP handling
- Invalid OTP with retry limit
- Phone number blocking after failures
- Token refresh flow

**Issue Creation:**
- Valid issue with all input types
- Issue with only image
- Issue with only audio
- Issue with only text
- Missing location handling
- Large file handling
- Invalid file format rejection

**AI Classification:**
- Image classification accuracy
- Audio transcription accuracy
- Text analysis entity extraction
- Severity detection correctness
- Duplicate detection precision/recall
- Spam detection false positive rate

**Status Workflow:**
- Valid status transitions
- Invalid status transition rejection
- Status history recording
- Notification triggering on status change

**Priority Calculation:**
- Correct weight application
- Score bounds (0-100)
- Time penalty accumulation
- Duplicate count impact

**Geospatial Queries:**
- Issues within radius
- Heatmap generation
- Area score calculation

**Notifications:**
- Email delivery
- SMS delivery
- Push notification delivery
- Notification batching
- Preference respect

**API Security:**
- Authentication requirement
- Authorization checks
- Rate limiting
- Input validation
- SQL injection prevention
- XSS prevention

### Performance Testing

**Load Test Scenarios:**
1. Normal load: 100 concurrent users, 1000 requests/minute
2. Peak load: 500 concurrent users, 5000 requests/minute
3. Stress test: Gradually increase to breaking point

**Performance Targets:**
- API response time p95 < 500ms
- Image classification < 2 seconds
- Audio transcription < 5 seconds
- Database queries < 100ms
- Page load time < 2 seconds

### CI/CD Pipeline

**Continuous Integration:**
1. Run linters (eslint, pylint, black)
2. Run unit tests
3. Run property-based tests
4. Check code coverage
5. Run security scans (Snyk, Bandit)
6. Build Docker images

**Continuous Deployment:**
1. Deploy to staging environment
2. Run integration tests
3. Run smoke tests
4. Deploy to production (blue-green)
5. Run health checks
6. Monitor error rates

## Deployment Architecture

### Container Structure

**Frontend Container:**
- Base: Node.js 18 Alpine
- Build: React production build
- Serve: Nginx
- Size: ~50MB

**Backend API Container:**
- Base: Python 3.10 Slim
- Framework: FastAPI with Uvicorn
- Workers: 4 per container
- Size: ~200MB

**AI Service Container:**
- Base: Python 3.10 with CUDA (for GPU)
- Models: Loaded at startup
- GPU: Optional, falls back to CPU
- Size: ~2GB (includes models)

**Worker Container:**
- Base: Python 3.10 Slim
- Purpose: Celery workers for async tasks
- Queues: notifications, ai_processing, analytics
- Size: ~200MB

### Kubernetes Deployment

**Namespaces:**
- `production` - Production environment
- `staging` - Staging environment
- `monitoring` - Monitoring tools

**Deployments:**
- `frontend` - 3 replicas, autoscale to 10
- `api` - 5 replicas, autoscale to 20
- `ai-service` - 2 replicas (GPU), autoscale to 5
- `worker` - 3 replicas, autoscale to 10
- `redis` - 1 replica (StatefulSet)
- `postgres` - 1 primary + 2 read replicas (StatefulSet)

**Services:**
- `frontend-service` - LoadBalancer
- `api-service` - ClusterIP
- `ai-service` - ClusterIP
- `redis-service` - ClusterIP
- `postgres-service` - ClusterIP

**Ingress:**
- TLS termination
- Path-based routing
- Rate limiting
- CORS configuration

**Autoscaling:**
- Horizontal Pod Autoscaler (HPA)
- Metrics: CPU > 70%, Memory > 80%
- Scale up: Add 2 pods
- Scale down: Remove 1 pod every 5 minutes

### Database Deployment

**PostgreSQL Configuration:**
- Version: 15+
- Extensions: PostGIS for geospatial
- Replication: 1 primary + 2 read replicas
- Backup: Daily full backup, continuous WAL archiving
- Retention: 30 days

**Redis Configuration:**
- Version: 7+
- Mode: Cluster mode for high availability
- Persistence: RDB snapshots + AOF
- Memory: 8GB per node
- Eviction: LRU policy

### Cloud Storage

**Media Storage:**
- Service: AWS S3 or Google Cloud Storage
- Buckets: `issues-media-prod`, `issues-media-archive`
- Lifecycle: Move to cold storage after 90 days
- CDN: CloudFront or Cloud CDN
- Access: Signed URLs with 1-hour expiry

### Monitoring and Observability

**Metrics Collection:**
- Prometheus for metrics
- Grafana for visualization
- Custom dashboards for business metrics

**Logging:**
- Centralized logging with ELK stack or Cloud Logging
- Structured JSON logs
- Log retention: 30 days

**Tracing:**
- Distributed tracing with Jaeger or Cloud Trace
- Trace sampling: 10% of requests

**Alerting:**
- PagerDuty or Opsgenie for on-call
- Slack integration for non-critical alerts

### Security Measures

**Network Security:**
- VPC with private subnets for databases
- Security groups restricting access
- WAF for DDoS protection
- API Gateway for rate limiting

**Data Security:**
- Encryption at rest (AES-256)
- Encryption in transit (TLS 1.3)
- Phone numbers encrypted in database
- Secrets managed with Vault or Secret Manager

**Authentication & Authorization:**
- JWT tokens with 1-hour expiry
- Refresh tokens with 30-day expiry
- Role-based access control (RBAC)
- API key authentication for external integrations

**Compliance:**
- GDPR compliance for data privacy
- Data retention policies
- User data export capability
- Right to be forgotten implementation

### Disaster Recovery

**Backup Strategy:**
- Database: Daily full backup + continuous WAL
- Media files: Cross-region replication
- Configuration: Version controlled in Git

**Recovery Objectives:**
- RTO (Recovery Time Objective): 4 hours
- RPO (Recovery Point Objective): 1 hour

**Disaster Recovery Plan:**
1. Detect failure through monitoring
2. Assess impact and decide on recovery
3. Failover to backup region if needed
4. Restore from latest backup
5. Verify data integrity
6. Resume operations
7. Post-mortem and improvements


## Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

### Property Reflection

After analyzing all acceptance criteria, I identified several areas where properties can be consolidated to avoid redundancy:

**Priority Score Calculation (6.2-6.8)**: Rather than testing each weight component separately, we can combine these into comprehensive properties that test the overall formula and bounds.

**Status Workflow (9.1-9.7)**: The individual status transitions can be combined into properties about valid state transitions and audit trail completeness.

**Notification Routing (11.2-11.4)**: These can be combined into a single property about preference-based routing.

**Severity Weight Assignment (6.2-6.5)**: These are all part of the same mapping and can be tested together.

**Database Schema (26.1-26.7)**: These are design requirements, not functional properties, so they don't need separate properties.

### Authentication Properties

**Property 1: OTP Generation for Valid Phone Numbers**
*For any* valid phone number format, when a user requests an OTP, the system should generate and send an OTP to that number.
**Validates: Requirements 1.1**

**Property 2: Valid OTP Authentication**
*For any* valid OTP submitted within the 5-minute expiry window, the system should successfully authenticate the user and create a session.
**Validates: Requirements 1.2**

**Property 3: OTP Expiry Enforcement**
*For any* OTP older than 5 minutes, the system should reject authentication attempts regardless of OTP correctness.
**Validates: Requirements 1.3**

**Property 4: Retry Limit Enforcement**
*For any* phone number, after 3 consecutive failed OTP attempts, the system should block that phone number for 15 minutes.
**Validates: Requirements 1.4, 1.5**

### Input Validation Properties

**Property 5: Image File Validation**
*For any* image file, the system should accept it if and only if it is in JPEG, PNG, or WebP format and under 10MB in size.
**Validates: Requirements 2.1**

**Property 6: Audio Duration Validation**
*For any* audio recording, the system should accept it if and only if its duration is 2 minutes or less.
**Validates: Requirements 2.2**

**Property 7: Text Length Validation**
*For any* text input, the system should accept it if and only if it is 1000 characters or less.
**Validates: Requirements 2.3**

**Property 8: Minimum Input Requirement**
*For any* issue submission, the system should accept it if and only if at least one input method (photo, voice, or text) is provided.
**Validates: Requirements 2.4**

### AI Classification Properties

**Property 9: Image Analysis Trigger**
*For any* uploaded image, the system should trigger AI classification analysis.
**Validates: Requirements 3.1**

**Property 10: Classification Category Domain**
*For any* image classification result, the predicted category should be one of: Road, Water, Garbage, Electricity, Safety, Sanitation, Drainage, or Streetlight.
**Validates: Requirements 3.2**

**Property 11: Confidence Score Bounds**
*For any* AI classification, the confidence score should be between 0 and 1 (inclusive).
**Validates: Requirements 3.3**

**Property 12: Low Confidence Flagging**
*For any* classification with confidence score below 0.6, the system should flag the issue for manual verification.
**Validates: Requirements 3.4**

**Property 13: Maximum Confidence Selection**
*For any* image with multiple object detections, the system should select the category with the highest confidence score.
**Validates: Requirements 3.5**

**Property 14: Voice Transcription Trigger**
*For any* voice recording submission, the system should trigger speech-to-text conversion.
**Validates: Requirements 4.1**

**Property 15: Text Analysis Trigger**
*For any* text input (from transcription or direct entry), the system should trigger NLP analysis for category identification.
**Validates: Requirements 4.2**

**Property 16: Entity Extraction Completeness**
*For any* completed NLP analysis, the system should extract location names, issue types, and urgency indicators from the text.
**Validates: Requirements 4.3**

**Property 17: Dominant Language Processing**
*For any* multi-lingual audio recording, the system should process the dominant language detected in the recording.
**Validates: Requirements 4.5**

### Severity Detection Properties

**Property 18: Severity Domain Constraint**
*For any* analyzed issue, the assigned severity should be one of: Low, Medium, High, or Critical.
**Validates: Requirements 5.1**

**Property 19: Safety Keyword Critical Assignment**
*For any* issue containing safety-related keywords (fire, accident, injury, danger), the system should assign Critical severity.
**Validates: Requirements 5.2**

**Property 20: Infrastructure Failure High Assignment**
*For any* issue containing infrastructure failure keywords (major leak, road collapse, power outage), the system should assign High severity.
**Validates: Requirements 5.3**

**Property 21: Maintenance Keyword Medium/Low Assignment**
*For any* issue containing maintenance keywords (minor damage, litter, dim light), the system should assign Medium or Low severity.
**Validates: Requirements 5.4**

**Property 22: Image Hazard Override**
*For any* issue where image analysis detects hazardous conditions, the system should assign Critical severity regardless of text-based severity.
**Validates: Requirements 5.5**

### Priority Scoring Properties

**Property 23: Priority Score Formula Application**
*For any* created issue, the priority score should equal the sum of severity weight, location risk weight, crowd reports weight, and time pending weight.
**Validates: Requirements 6.1**

**Property 24: Severity Weight Mapping**
*For any* issue, the severity weight should be: 40 for Critical, 30 for High, 20 for Medium, or 10 for Low.
**Validates: Requirements 6.2, 6.3, 6.4, 6.5**

**Property 25: Location Risk Bounds**
*For any* issue, the location risk weight should be between 0 and 20 (inclusive).
**Validates: Requirements 6.6**

**Property 26: Crowd Reports Scoring**
*For any* issue with duplicate reports, the crowd reports weight should be 5 points per additional report, capped at 25 points maximum.
**Validates: Requirements 6.7**

**Property 27: Time Penalty Accumulation**
*For any* unresolved issue, the time pending weight should be 1 point per day since creation, capped at 15 points maximum.
**Validates: Requirements 6.8**

**Property 28: Priority Score Bounds**
*For any* issue, the total priority score should be between 10 and 100 (inclusive), given the component bounds.
**Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8**

### Duplicate Detection Properties

**Property 29: Geospatial Duplicate Search**
*For any* new issue submission, the system should search for similar issues within a 500-meter radius.
**Validates: Requirements 7.1**

**Property 30: Duplicate Detection Criteria**
*For any* new issue, if an existing unresolved issue exists with the same category, within 500 meters, and submitted within 24 hours, the system should mark the new issue as a duplicate.
**Validates: Requirements 7.2**

**Property 31: Duplicate Linking and Counting**
*For any* issue marked as duplicate, the system should link it to the original issue and increment the original's crowd report count by 1.
**Validates: Requirements 7.3**

**Property 32: Duplicate Notification**
*For any* detected duplicate, the system should notify the submitting user and provide the original issue's tracking information.
**Validates: Requirements 7.4**

**Property 33: Cascade Resolution**
*For any* original issue that is resolved, all linked duplicate issues should also be marked as resolved.
**Validates: Requirements 7.5**

### Spam Detection Properties

**Property 34: Spam Analysis Trigger**
*For any* submitted issue, the system should analyze it for spam indicators.
**Validates: Requirements 8.1**

**Property 35: High Confidence Spam Flagging**
*For any* issue with spam detection confidence above 0.8, the system should flag it as potential spam.
**Validates: Requirements 8.2**

**Property 36: Rate-Based Spam Detection**
*For any* user who submits more than 5 issues within 10 minutes, subsequent submissions should be flagged for manual review.
**Validates: Requirements 8.3**

**Property 37: Spam Visibility Control**
*For any* issue flagged as spam, the system should prevent it from appearing in public dashboards until verified.
**Validates: Requirements 8.4**

**Property 38: Trust Score Update on Fake Marking**
*For any* issue marked as fake by authorities, the system should update the submitting user's trust score.
**Validates: Requirements 8.5**

### Status Workflow Properties

**Property 39: Initial Status Assignment**
*For any* newly created issue, the system should set its status to Submitted.
**Validates: Requirements 9.1**

**Property 40: Valid Status Transitions**
*For any* issue, status transitions should follow the valid workflow: Submitted → Verified → Assigned → In Progress → Resolved.
**Validates: Requirements 9.2, 9.3, 9.4, 9.5**

**Property 41: Status Change Audit Trail**
*For any* status change, the system should record the timestamp, the user who made the change, and optional notes in the status history.
**Validates: Requirements 9.6**

**Property 42: Resolution Evidence Requirement**
*For any* issue transitioning to Resolved status, the system should require photo evidence of completion.
**Validates: Requirements 9.7**

### Assignment Properties

**Property 43: Category-Based Department Assignment**
*For any* verified issue, the system should allow assignment to a department that handles the issue's category.
**Validates: Requirements 10.1**

**Property 44: Assignment Notification**
*For any* issue assignment, the system should send notifications to the assigned department via email and dashboard.
**Validates: Requirements 10.2**

**Property 45: Capacity Warning**
*For any* department with more than 50 active issues, the system should warn administrators when attempting new assignments.
**Validates: Requirements 10.3**

**Property 46: Unassigned Issue Escalation**
*For any* verified issue that remains unassigned for 24 hours, the system should auto-escalate it to the supervisor.
**Validates: Requirements 10.4**

**Property 47: Reassignment Tracking**
*For any* issue reassignment, the system should record the reassignment reason and notify both the old and new departments.
**Validates: Requirements 10.5**

### Notification Properties

**Property 48: Status Change Notification Trigger**
*For any* issue status change, the system should send a notification to the reporting user.
**Validates: Requirements 11.1**

**Property 49: Preference-Based Notification Routing**
*For any* notification, the system should send it via email if email is enabled, via SMS if SMS is enabled, and via web push if the user is logged in.
**Validates: Requirements 11.2, 11.3, 11.4**

**Property 50: Resolution Notification Content**
*For any* resolved issue, the system should send a final notification containing resolution details and photo evidence.
**Validates: Requirements 11.5**

### Escalation Properties

**Property 51: Critical Issue Time Escalation**
*For any* Critical severity issue that remains in Submitted status for 2 hours, the system should escalate it to the department head.
**Validates: Requirements 12.1**

**Property 52: High Issue Time Escalation**
*For any* High severity issue that remains in Assigned status for 24 hours, the system should escalate it to the supervisor.
**Validates: Requirements 12.2**

**Property 53: Stalled Issue Escalation**
*For any* issue that remains in In Progress status for 7 days, the system should escalate it to senior management.
**Validates: Requirements 12.3**

**Property 54: Escalation Notification Delivery**
*For any* escalated issue, the system should send notifications to all designated escalation recipients.
**Validates: Requirements 12.4**

**Property 55: Escalation Recording in Resolution**
*For any* escalated issue that is resolved, the system should record the escalation details in the resolution report.
**Validates: Requirements 12.5**

### Emergency Mode Properties

**Property 56: Emergency Mode Severity Assignment**
*For any* issue submitted in emergency mode, the system should automatically assign Critical severity.
**Validates: Requirements 13.1**

**Property 57: Emergency Workflow Bypass**
*For any* emergency issue, the system should bypass normal verification and immediately assign it to the relevant department.
**Validates: Requirements 13.2**

**Property 58: Emergency Alert Delivery**
*For any* emergency issue creation, the system should send immediate SMS and email alerts to on-duty personnel.
**Validates: Requirements 13.3**

**Property 59: Emergency Visual Indicator**
*For any* emergency issue, the Authority Dashboard should display it with a red alert indicator.
**Validates: Requirements 13.4**

**Property 60: Emergency Backup Escalation**
*For any* emergency issue that remains unacknowledged for 15 minutes, the system should send alerts to backup contacts.
**Validates: Requirements 13.5**

### Map Visualization Properties

**Property 61: Active Issues Map Display**
*For any* Authority Dashboard load, all active issues should be displayed as pins on the map.
**Validates: Requirements 14.1**

**Property 62: Category Filter Application**
*For any* category filter selection, the map should update to show only issues matching that category.
**Validates: Requirements 14.3**

**Property 63: Severity Color Coding**
*For any* issue displayed on the map, the pin color should be: red for Critical, orange for High, yellow for Medium, and green for Low.
**Validates: Requirements 14.4**

### Filtering and Search Properties

**Property 64: Multi-Criteria Filtering Support**
*For any* filter application, the system should support filtering by category, severity, status, date range, and geographic area.
**Validates: Requirements 15.1**

**Property 65: AND Logic Filter Combination**
*For any* multiple filter selection, the system should apply all filters with AND logic (intersection of results).
**Validates: Requirements 15.2**

**Property 66: Multi-Field Text Search**
*For any* text search query, the system should search across issue descriptions, locations, and IDs.
**Validates: Requirements 15.3**

**Property 67: Filter Result Consistency**
*For any* applied filters, the issue count and map visualization should match the filtered results.
**Validates: Requirements 15.4**

**Property 68: Filter Clear Restoration**
*For any* filter clear action, the system should restore and display the complete unfiltered issue list.
**Validates: Requirements 15.5**

### Time Tracking Properties

**Property 69: Submission Timestamp Recording**
*For any* newly created issue, the system should record the submission timestamp.
**Validates: Requirements 16.1**

**Property 70: Transition Timestamp Recording**
*For any* status transition, the system should record the timestamp of the change.
**Validates: Requirements 16.2**

**Property 71: Resolution Time Calculation**
*For any* resolved issue, the total resolution time should equal the difference between resolution timestamp and submission timestamp.
**Validates: Requirements 16.3**

**Property 72: Average Resolution Time Calculation**
*For any* department and category combination, the displayed average resolution time should equal the mean of all resolution times for that combination.
**Validates: Requirements 16.4**

**Property 73: SLA Threshold Flagging**
*For any* issue, if resolution time exceeds the SLA threshold for its severity (24h for Critical, 72h for High, 7d for Medium, 14d for Low), the system should flag it as overdue.
**Validates: Requirements 16.5**

### Analytics Properties

**Property 74: Daily Category Count Aggregation**
*For any* date and category, the displayed issue count should equal the number of issues created on that date in that category.
**Validates: Requirements 17.1**

**Property 75: Geographic Density Heatmap**
*For any* geographic area, the heatmap intensity should reflect the density of issues in that area.
**Validates: Requirements 17.2**

**Property 76: Department Resolution Rate Calculation**
*For any* department, the resolution rate percentage should equal (resolved issues / total issues) × 100.
**Validates: Requirements 17.3**

**Property 77: Time Trend Calculation**
*For any* time period, the displayed average response and resolution times should reflect the mean values for that period.
**Validates: Requirements 17.4**

**Property 78: Date Range Analytics Filtering**
*For any* selected date range, all analytics should include only data from that period.
**Validates: Requirements 17.5**

**Property 79: Report Export Completeness**
*For any* report export, the generated CSV or PDF file should contain all selected metrics in the specified format.
**Validates: Requirements 17.6**

### Public Dashboard Properties

**Property 80: Public Aggregate Counts**
*For any* Public Portal access, the displayed totals for reported, resolved, and in-progress issues should match the actual counts.
**Validates: Requirements 18.1**

**Property 81: Area Resolution Time Averages**
*For any* ward or district, the displayed average resolution time should equal the mean resolution time for all issues in that area.
**Validates: Requirements 18.2**

**Property 82: Ward Performance Leaderboard**
*For any* leaderboard display, wards should be ranked by resolution rate in descending order.
**Validates: Requirements 18.3**

**Property 83: Area Cleanliness Score Calculation**
*For any* area, the cleanliness score should be calculated based on issue density and resolution speed according to the defined formula.
**Validates: Requirements 18.4**

**Property 84: Public Map Anonymization**
*For any* issue displayed on the public map, no personal information (reporter identity, contact details) should be visible.
**Validates: Requirements 18.5**

### User Issue Tracking Properties

**Property 85: User Issue Filtering**
*For any* logged-in citizen, the displayed issues should include all and only issues reported by that user.
**Validates: Requirements 19.1**

**Property 86: Issue Detail Completeness**
*For any* issue view, the display should include current status, assigned department, and complete status history.
**Validates: Requirements 19.2**

**Property 87: Resolution Time Estimation**
*For any* issue view, the system should display an estimated resolution time based on the issue's category and severity.
**Validates: Requirements 19.4**

**Property 88: Resolution Photo Display**
*For any* resolved issue, the display should include both before and after photos.
**Validates: Requirements 19.5**

### Security and Privacy Properties

**Property 89: Phone Number Encryption**
*For any* user registration, the stored phone number should be in encrypted format.
**Validates: Requirements 20.1**

**Property 90: Public Display Privacy**
*For any* issue displayed on the Public Portal, reporter identity and contact information should not be revealed.
**Validates: Requirements 20.2**

**Property 91: Authorized Contact Access**
*For any* issue viewed by authority personnel, reporter contact information should be visible only to authorized users.
**Validates: Requirements 20.3**

**Property 92: Data Deletion Anonymization**
*For any* user data deletion request, all reports should be anonymized (personal data removed) while preserving issue data.
**Validates: Requirements 20.4**

**Property 93: API Authentication Requirement**
*For any* API request, the system should require a valid authentication token and validate permissions before processing.
**Validates: Requirements 20.5**

### Media Storage Properties

**Property 94: Image Compression**
*For any* uploaded image, the system should compress it to reduce file size while maintaining readability.
**Validates: Requirements 21.1**

**Property 95: Audio Format Conversion**
*For any* uploaded audio file, the system should convert it to a compressed format (MP3 or Opus).
**Validates: Requirements 21.2**

**Property 96: Media Archival**
*For any* issue resolved for more than 90 days, the associated media files should be moved to cold storage.
**Validates: Requirements 21.4**

**Property 97: Signed URL Generation**
*For any* media file request, the system should generate a time-limited signed URL (expiring within specified duration).
**Validates: Requirements 21.5**

### AI Model Monitoring Properties

**Property 98: Classification Logging**
*For any* AI classification, the system should log the prediction, confidence score, and model version.
**Validates: Requirements 22.1**

**Property 99: Correction Recording**
*For any* manual correction of AI classification, the system should record the correction as ground truth data.
**Validates: Requirements 22.2**

**Property 100: Accuracy Calculation**
*For any* category, the displayed classification accuracy should equal (correct predictions / total predictions) × 100.
**Validates: Requirements 22.3**

**Property 101: Accuracy Alert Threshold**
*For any* category where accuracy drops below 80%, the system should alert administrators.
**Validates: Requirements 22.4**

**Property 102: Retraining Flag Threshold**
*For any* model with 1000 or more correction samples collected, the system should flag it for retraining.
**Validates: Requirements 22.5**

### Offline Capability Properties

**Property 103: Offline Draft Creation**
*For any* detected loss of internet connection, the system should allow users to create draft reports.
**Validates: Requirements 24.1**

**Property 104: Local Draft Storage**
*For any* draft created offline, the system should store it in browser local storage.
**Validates: Requirements 24.2**

**Property 105: Reconnection Draft Prompt**
*For any* internet connectivity restoration, the system should prompt the user to submit any pending drafts.
**Validates: Requirements 24.3**

**Property 106: Draft Submission Completeness**
*For any* submitted draft, the system should upload all associated media files and create the complete issue.
**Validates: Requirements 24.4**

### API Properties

**Property 107: JWT Authentication Requirement**
*For any* API endpoint access, the system should require a valid JWT token.
**Validates: Requirements 25.1**

**Property 108: JSON Response Format**
*For any* API request, the response should be in JSON format with consistent structure.
**Validates: Requirements 25.3**

**Property 109: Error Response Format**
*For any* API error, the response should include an appropriate HTTP status code and error message.
**Validates: Requirements 25.4**

**Property 110: Rate Limit Enforcement**
*For any* user exceeding 100 requests per minute, subsequent requests should return HTTP 429 status code.
**Validates: Requirements 25.5**

### System Monitoring Properties

**Property 111: Error Logging**
*For any* system error, the system should log the error details to the centralized logging service.
**Validates: Requirements 27.7**

