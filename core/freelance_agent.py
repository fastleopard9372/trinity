import json
import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from openai import OpenAI
import hashlib

from config.settings import settings
from utils.logger import logger
from clients.nas_client import NASClient

class JobStatus(Enum):
    NEW = "new"
    ANALYZING = "analyzing"
    PROPOSAL_GENERATED = "proposal_generated"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    ARCHIVED = "archived"

@dataclass
class JobListing:
    id: str
    title: str
    description: str
    client: str
    budget: Optional[str]
    skills_required: List[str]
    posted_date: datetime
    deadline: Optional[datetime]
    url: Optional[str]
    platform: str
    status: JobStatus = JobStatus.NEW
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class JobProposal:
    job_id: str
    proposal_text: str
    pricing: str
    timeline: str
    key_points: List[str]
    generated_at: datetime
    reviewed: bool = False
    approved: bool = False
    sent: bool = False
    reviewer_notes: Optional[str] = None

class JobFeedManager:
    """Manages mock job feed for testing"""
    
    def __init__(self):
        self.nas_client = NASClient()
        self.feed_path = f"{settings.nas_base_path}/freelance/job_feed.json"
        self.jobs_cache = []
        
    def create_mock_job_feed(self):
        """Create mock job feed with sample jobs"""
        mock_jobs = [
            {
                "id": "job_001",
                "title": "Full Stack Developer for E-commerce Platform",
                "description": "We need an experienced full stack developer to build a modern e-commerce platform using React and Node.js. The project includes user authentication, payment integration, inventory management, and admin dashboard. Must have experience with PostgreSQL and AWS deployment.",
                "client": "TechStartup Inc",
                "budget": "$5000-8000",
                "skills_required": ["React", "Node.js", "PostgreSQL", "AWS", "Payment Integration"],
                "posted_date": datetime.now().isoformat(),
                "deadline": (datetime.now() + timedelta(days=30)).isoformat(),
                "url": "https://example.com/job/001",
                "platform": "Upwork"
            },
            {
                "id": "job_002", 
                "title": "Python Data Analysis and Visualization",
                "description": "Looking for a Python expert to analyze sales data and create interactive dashboards. You'll work with pandas, matplotlib, and build a Streamlit app for data visualization. Must have experience with statistical analysis and data cleaning.",
                "client": "DataCorp Solutions",
                "budget": "$2000-3500",
                "skills_required": ["Python", "Pandas", "Data Analysis", "Streamlit", "Statistics"],
                "posted_date": datetime.now().isoformat(),
                "deadline": (datetime.now() + timedelta(days=14)).isoformat(),
                "url": "https://example.com/job/002",
                "platform": "Freelancer"
            },
            {
                "id": "job_003",
                "title": "AI Chatbot Development with FastAPI",
                "description": "Need to develop an intelligent chatbot using FastAPI backend and integrate it with OpenAI GPT. The bot should handle customer service queries, have conversation memory, and integrate with our CRM system. Experience with vector databases preferred.",
                "client": "CustomerFirst LLC",
                "budget": "$3000-5000",
                "skills_required": ["FastAPI", "OpenAI API", "Vector Databases", "NLP", "Python"],
                "posted_date": (datetime.now() - timedelta(hours=2)).isoformat(),
                "deadline": (datetime.now() + timedelta(days=21)).isoformat(),
                "url": "https://example.com/job/003",
                "platform": "Upwork"
            }
        ]
        
        # Save to NAS
        temp_file = "temp/job_feed.json"
        os.makedirs("temp", exist_ok=True)
        
        with open(temp_file, 'w') as f:
            json.dump(mock_jobs, f, indent=2)
        
        if self.nas_client.upload_file(temp_file, self.feed_path):
            logger.info("Mock job feed created successfully")
            os.remove(temp_file)
            return True
        else:
            logger.error("Failed to upload job feed to NAS")
            return False
    
    def fetch_jobs(self) -> List[JobListing]:
        """Fetch jobs from the feed"""
        try:
            # Download from NAS
            temp_file = "temp/downloaded_feed.json"
            os.makedirs("temp", exist_ok=True)
            
            if self.nas_client.download_file(self.feed_path, temp_file):
                with open(temp_file, 'r') as f:
                    jobs_data = json.load(f)
                
                jobs = []
                for job_data in jobs_data:
                    job = JobListing(
                        id=job_data['id'],
                        title=job_data['title'],
                        description=job_data['description'],
                        client=job_data['client'],
                        budget=job_data.get('budget'),
                        skills_required=job_data['skills_required'],
                        posted_date=datetime.fromisoformat(job_data['posted_date']),
                        deadline=datetime.fromisoformat(job_data['deadline']) if job_data.get('deadline') else None,
                        url=job_data.get('url'),
                        platform=job_data['platform']
                    )
                    jobs.append(job)
                
                os.remove(temp_file)
                self.jobs_cache = jobs
                logger.info(f"Fetched {len(jobs)} jobs from feed")
                return jobs
            else:
                logger.error("Failed to download job feed from NAS")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching jobs: {e}")
            return []

class ProposalGenerator:
    """Generates job proposals using templates and AI"""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict:
        """Load proposal templates"""
        return {
            "intro": [
                "Hi {client}, I'm excited about your {title} project.",
                "Hello {client}, I've reviewed your {title} requirement and I'm confident I can deliver excellent results.",
                "Dear {client}, Your {title} project aligns perfectly with my expertise."
            ],
            "experience": [
                "I have extensive experience in {skills} with {years}+ years in the field.",
                "My background includes strong expertise in {skills}, having completed similar projects successfully.",
                "I specialize in {skills} and have delivered numerous projects in this domain."
            ],
            "approach": [
                "My approach would involve: 1) Requirements analysis, 2) Technical design, 3) Iterative development, 4) Testing and deployment.",
                "I propose a structured approach: initial consultation, detailed planning, agile development, and thorough testing.",
                "I'll start with a comprehensive analysis, followed by milestone-based development with regular updates."
            ],
            "closing": [
                "I'd love to discuss this project further. Let's schedule a call to align on requirements and timelines.",
                "I'm ready to start immediately and can deliver within your timeline. Looking forward to collaborating.",
                "Please feel free to review my portfolio and let's discuss how I can help bring your vision to life."
            ]
        }
    
    def analyze_job_fit(self, job: JobListing) -> Dict:
        """Analyze how well the job fits our capabilities"""
        
        # Define our skill set
        our_skills = {
            "python": 9,
            "javascript": 8,
            "react": 8,
            "node.js": 8,
            "fastapi": 9,
            "postgresql": 7,
            "aws": 7,
            "data analysis": 8,
            "machine learning": 7,
            "ai": 8,
            "openai api": 9,
            "streamlit": 8,
            "pandas": 8,
            "api development": 9,
            "full stack": 8,
            "chatbot": 8,
            "nlp": 7
        }
        
        # Calculate match score
        matched_skills = []
        skill_scores = []
        
        for required_skill in job.skills_required:
            for our_skill, score in our_skills.items():
                if our_skill.lower() in required_skill.lower():
                    matched_skills.append(required_skill)
                    skill_scores.append(score)
                    break
        
        # Calculate overall fit score
        if matched_skills:
            avg_skill_score = sum(skill_scores) / len(skill_scores)
            coverage_score = len(matched_skills) / len(job.skills_required)
            overall_fit = (avg_skill_score * 0.7 + coverage_score * 10 * 0.3)
        else:
            overall_fit = 0
        
        # Analyze budget
        budget_analysis = self._analyze_budget(job.budget)
        
        return {
            "overall_fit_score": round(overall_fit, 2),
            "matched_skills": matched_skills,
            "missing_skills": [skill for skill in job.skills_required if skill not in matched_skills],
            "skill_scores": dict(zip(matched_skills, skill_scores)),
            "budget_analysis": budget_analysis,
            "recommendation": self._get_recommendation(overall_fit, budget_analysis)
        }
    
    def _analyze_budget(self, budget_str: Optional[str]) -> Dict:
        """Analyze budget information"""
        if not budget_str:
            return {"status": "unknown", "min": None, "max": None, "suitable": None}
        
        # Extract numbers from budget string
        numbers = re.findall(r'\$?(\d+(?:,\d{3})*)', budget_str)
        if numbers:
            amounts = [int(n.replace(',', '')) for n in numbers]
            min_budget = min(amounts)
            max_budget = max(amounts) if len(amounts) > 1 else min_budget
            
            # Our minimum acceptable rate (configurable)
            min_acceptable = 1000
            
            return {
                "status": "parsed",
                "min": min_budget,
                "max": max_budget,
                "suitable": max_budget >= min_acceptable,
                "our_min_rate": min_acceptable
            }
        
        return {"status": "unparseable", "original": budget_str, "suitable": None}
    
    def _get_recommendation(self, fit_score: float, budget_analysis: Dict) -> str:
        """Get recommendation based on analysis"""
        if fit_score >= 7 and budget_analysis.get("suitable", True):
            return "HIGHLY_RECOMMENDED"
        elif fit_score >= 5 and budget_analysis.get("suitable", True):
            return "RECOMMENDED"
        elif fit_score >= 3:
            return "CONSIDER"
        else:
            return "SKIP"
    
    def generate_proposal(self, job: JobListing, analysis: Dict) -> JobProposal:
        """Generate a proposal for the job"""
        
        if self.client and analysis["recommendation"] in ["HIGHLY_RECOMMENDED", "RECOMMENDED"]:
            # Use AI for high-potential jobs
            proposal_text = self._generate_ai_proposal(job, analysis)
        else:
            # Use template for others
            proposal_text = self._generate_template_proposal(job, analysis)
        
        # Extract pricing and timeline
        pricing = self._generate_pricing(job, analysis)
        timeline = self._generate_timeline(job)
        key_points = self._extract_key_points(job, analysis)
        
        return JobProposal(
            job_id=job.id,
            proposal_text=proposal_text,
            pricing=pricing,
            timeline=timeline,
            key_points=key_points,
            generated_at=datetime.now()
        )
    
    def _generate_ai_proposal(self, job: JobListing, analysis: Dict) -> str:
        """Generate AI-powered proposal"""
        try:
            prompt = f"""
Write a professional freelance proposal for this job:

Job Title: {job.title}
Client: {job.client}
Description: {job.description}
Required Skills: {', '.join(job.skills_required)}
Budget: {job.budget}

My Analysis:
- Matched Skills: {', '.join(analysis['matched_skills'])}
- Fit Score: {analysis['overall_fit_score']}/10
- Recommendation: {analysis['recommendation']}

Write a compelling 200-300 word proposal that:
1. Shows enthusiasm and understanding of the project
2. Highlights relevant experience with matched skills
3. Outlines a clear approach
4. Demonstrates value and professionalism
5. Includes a call to action

Keep it personal, confident, and focused on the client's needs.
"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert freelancer writing winning proposals. Be professional, confident, and client-focused."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"AI proposal generation failed: {e}")
            return self._generate_template_proposal(job, analysis)
    
    def _generate_template_proposal(self, job: JobListing, analysis: Dict) -> str:
        """Generate template-based proposal"""
        import random
        
        # Select random templates
        intro = random.choice(self.templates["intro"]).format(
            client=job.client,
            title=job.title
        )
        
        experience = random.choice(self.templates["experience"]).format(
            skills=", ".join(analysis['matched_skills'][:3]),
            years=5  # Could be made configurable
        )
        
        approach = random.choice(self.templates["approach"])
        closing = random.choice(self.templates["closing"])
        
        # Add specific project understanding
        project_understanding = f"""
I understand you need {job.title.lower()} with focus on {', '.join(job.skills_required[:3])}. 
Based on your requirements, I can deliver a solution that meets your specifications and timeline.
"""
        
        proposal = f"""
{intro}

{project_understanding}

{experience}

{approach}

{closing}

Best regards,
Trinity AI Assistant
"""
        
        return proposal.strip()
    
    def _generate_pricing(self, job: JobListing, analysis: Dict) -> str:
        """Generate pricing strategy"""
        budget_analysis = analysis["budget_analysis"]
        
        if budget_analysis["status"] == "parsed":
            # Price competitively within range
            min_budget = budget_analysis["min"]
            max_budget = budget_analysis["max"]
            
            if analysis["overall_fit_score"] >= 7:
                # High fit - price at 80-90% of max
                our_price = int(max_budget * 0.85)
            else:
                # Medium fit - price at 60-70% of max
                our_price = int(max_budget * 0.65)
            
            return f"${our_price:,} for the complete project"
        else:
            # No budget info - provide range
            return "$2,000 - $5,000 depending on specific requirements"
    
    def _generate_timeline(self, job: JobListing) -> str:
        """Generate timeline estimate"""
        # Simple heuristic based on job complexity
        if any(keyword in job.description.lower() for keyword in ["complex", "large", "enterprise", "multiple"]):
            return "4-6 weeks for full delivery with weekly milestones"
        elif any(keyword in job.description.lower() for keyword in ["simple", "basic", "small"]):
            return "1-2 weeks for complete implementation"
        else:
            return "2-4 weeks with milestone-based delivery"
    
    def _extract_key_points(self, job: JobListing, analysis: Dict) -> List[str]:
        """Extract key selling points"""
        points = []
        
        # Add skill matches
        if analysis["matched_skills"]:
            points.append(f"Expert in {', '.join(analysis['matched_skills'][:3])}")
        
        # Add project understanding
        if "api" in job.description.lower():
            points.append("Experienced in API development and integration")
        
        if "dashboard" in job.description.lower():
            points.append("Skilled in creating intuitive dashboards and UIs")
        
        if "database" in job.description.lower():
            points.append("Strong database design and optimization skills")
        
        # Add delivery commitment
        points.append("Committed to on-time delivery and regular communication")
        points.append("Providing comprehensive documentation and support")
        
        return points[:5]  # Limit to top 5 points

class FreelanceAgent:
    """Main freelance agent orchestrator"""
    
    def __init__(self):
        self.job_feed_manager = JobFeedManager()
        self.proposal_generator = ProposalGenerator()
        self.nas_client = NASClient()
        
        # Create agent directories
        self._setup_agent_structure()
    
    def _setup_agent_structure(self):
        """Setup folder structure for freelance agent"""
        folders = [
            f"{settings.nas_base_path}/freelance",
            f"{settings.nas_base_path}/freelance/jobs",
            f"{settings.nas_base_path}/freelance/proposals",
            f"{settings.nas_base_path}/freelance/drafts",
            f"{settings.nas_base_path}/freelance/logs",
            f"{settings.nas_base_path}/freelance/analytics"
        ]
        
        for folder in folders:
            try:
                # Create folder structure on NAS
                if self.nas_client.fs:
                    self.nas_client.fs.create_folder(folder)
                logger.debug(f"Created freelance folder: {folder}")
            except Exception as e:
                logger.debug(f"Folder creation info for {folder}: {e}")
    
    def initialize_system(self):
        """Initialize the freelance agent system"""
        logger.info("Initializing Freelance Agent System...")
        
        # Setup folder structure
        self._setup_agent_structure()
        
        # Create mock job feed
        if self.job_feed_manager.create_mock_job_feed():
            logger.info("Mock job feed created successfully")
        else:
            logger.error("Failed to create mock job feed")
            return False
        
        logger.info("Freelance Agent System initialized successfully")
        return True
    
    def process_job_feed(self) -> Dict:
        """Process the job feed and generate proposals"""
        logger.info("Processing job feed...")
        
        # Fetch jobs
        jobs = self.job_feed_manager.fetch_jobs()
        if not jobs:
            return {"error": "No jobs found in feed"}
        
        results = {
            "total_jobs": len(jobs),
            "processed": [],
            "skipped": [],
            "errors": []
        }
        
        for job in jobs:
            try:
                # Analyze job fit
                analysis = self.proposal_generator.analyze_job_fit(job)
                
                # Generate proposal if recommended
                if analysis["recommendation"] in ["HIGHLY_RECOMMENDED", "RECOMMENDED", "CONSIDER"]:
                    proposal = self.proposal_generator.generate_proposal(job, analysis)
                    
                    # Save job and proposal data
                    self._save_job_data(job, analysis, proposal)
                    
                    results["processed"].append({
                        "job_id": job.id,
                        "title": job.title,
                        "client": job.client,
                        "fit_score": analysis["overall_fit_score"],
                        "recommendation": analysis["recommendation"],
                        "proposal_generated": True
                    })
                    
                    logger.info(f"Processed job {job.id}: {job.title} - {analysis['recommendation']}")
                else:
                    results["skipped"].append({
                        "job_id": job.id,
                        "title": job.title,
                        "reason": analysis["recommendation"]
                    })
                    
                    logger.info(f"Skipped job {job.id}: {analysis['recommendation']}")
                    
            except Exception as e:
                logger.error(f"Error processing job {job.id}: {e}")
                results["errors"].append({
                    "job_id": job.id,
                    "error": str(e)
                })
        
        # Save processing results
        self._save_processing_results(results)
        
        logger.info(f"Job feed processing complete: {len(results['processed'])} processed, {len(results['skipped'])} skipped")
        return results
    
    def _save_job_data(self, job: JobListing, analysis: Dict, proposal: JobProposal):
        """Save job analysis and proposal to NAS"""
        
        # Prepare job data
        job_data = {
            "job": asdict(job),
            "analysis": analysis,
            "proposal": asdict(proposal),
            "processed_at": datetime.now().isoformat()
        }
        
        # Convert datetime objects to strings for JSON serialization
        job_data["job"]["posted_date"] = job.posted_date.isoformat()
        job_data["job"]["deadline"] = job.deadline.isoformat() if job.deadline else None
        job_data["job"]["created_at"] = job.created_at.isoformat()
        job_data["proposal"]["generated_at"] = proposal.generated_at.isoformat()
        
        # Save to temp file first
        temp_file = f"temp/job_{job.id}.json"
        os.makedirs("temp", exist_ok=True)
        
        with open(temp_file, 'w') as f:
            json.dump(job_data, f, indent=2)
        
        # Upload to NAS
        nas_path = f"{settings.nas_base_path}/freelance/jobs/job_{job.id}.json"
        if self.nas_client.upload_file(temp_file, nas_path):
            logger.info(f"Saved job data: {nas_path}")
        else:
            logger.error(f"Failed to save job data for {job.id}")
        
        # Save proposal separately for easy access
        proposal_path = f"{settings.nas_base_path}/freelance/proposals/proposal_{job.id}.txt"
        temp_proposal = f"temp/proposal_{job.id}.txt"
        
        with open(temp_proposal, 'w') as f:
            f.write(f"Job: {job.title}\n")
            f.write(f"Client: {job.client}\n")
            f.write(f"Fit Score: {analysis['overall_fit_score']}/10\n")
            f.write(f"Recommendation: {analysis['recommendation']}\n")
            f.write(f"Generated: {proposal.generated_at}\n")
            f.write(f"Pricing: {proposal.pricing}\n")
            f.write(f"Timeline: {proposal.timeline}\n\n")
            f.write("=== PROPOSAL ===\n\n")
            f.write(proposal.proposal_text)
            f.write("\n\n=== KEY POINTS ===\n")
            for point in proposal.key_points:
                f.write(f"• {point}\n")
        
        if self.nas_client.upload_file(temp_proposal, proposal_path):
            logger.info(f"Saved proposal: {proposal_path}")
        
        # Cleanup temp files
        try:
            os.remove(temp_file)
            os.remove(temp_proposal)
        except:
            pass
    
    def _save_processing_results(self, results: Dict):
        """Save processing results summary"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Add summary statistics
        results["summary"] = {
            "timestamp": datetime.now().isoformat(),
            "success_rate": len(results["processed"]) / results["total_jobs"] if results["total_jobs"] > 0 else 0,
            "high_fit_jobs": len([j for j in results["processed"] if j.get("fit_score", 0) >= 7]),
            "recommendations_breakdown": {}
        }
        
        # Count recommendations
        for job in results["processed"]:
            rec = job.get("recommendation", "UNKNOWN")
            results["summary"]["recommendations_breakdown"][rec] = results["summary"]["recommendations_breakdown"].get(rec, 0) + 1
        
        # Save to temp file
        temp_file = f"temp/processing_results_{timestamp}.json"
        with open(temp_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Upload to NAS
        nas_path = f"{settings.nas_base_path}/freelance/logs/processing_results_{timestamp}.json"
        if self.nas_client.upload_file(temp_file, nas_path):
            logger.info(f"Saved processing results: {nas_path}")
        
        try:
            os.remove(temp_file)
        except:
            pass
    
    def get_pending_proposals(self) -> List[Dict]:
        """Get proposals that need manual review"""
        pending = []
        
        try:
            # List files in proposals directory
            files = self.nas_client.list_files(f"{settings.nas_base_path}/freelance/jobs")
            
            for file_info in files:
                if file_info.get('name', '').startswith('job_') and file_info.get('name', '').endswith('.json'):
                    # Download and check status
                    temp_file = f"temp/{file_info['name']}"
                    if self.nas_client.download_file(f"{settings.nas_base_path}/freelance/jobs/{file_info['name']}", temp_file):
                        with open(temp_file, 'r') as f:
                            job_data = json.load(f)
                        
                        proposal = job_data.get('proposal', {})
                        if not proposal.get('reviewed', False):
                            pending.append({
                                "job_id": job_data['job']['id'],
                                "title": job_data['job']['title'],
                                "client": job_data['job']['client'],
                                "fit_score": job_data['analysis']['overall_fit_score'],
                                "recommendation": job_data['analysis']['recommendation'],
                                "generated_at": proposal['generated_at'],
                                "file_path": file_info['name']
                            })
                        
                        os.remove(temp_file)
                        
        except Exception as e:
            logger.error(f"Error getting pending proposals: {e}")
        
        return sorted(pending, key=lambda x: x['fit_score'], reverse=True)
    
    def approve_proposal(self, job_id: str, reviewer_notes: str = "") -> bool:
        """Approve a proposal for sending"""
        return self._update_proposal_status(job_id, approved=True, reviewer_notes=reviewer_notes)
    
    def reject_proposal(self, job_id: str, reviewer_notes: str = "") -> bool:
        """Reject a proposal"""
        return self._update_proposal_status(job_id, approved=False, reviewer_notes=reviewer_notes)
    
    def _update_proposal_status(self, job_id: str, approved: bool, reviewer_notes: str) -> bool:
        """Update proposal review status"""
        try:
            file_name = f"job_{job_id}.json"
            nas_file_path = f"{settings.nas_base_path}/freelance/jobs/{file_name}"
            temp_file = f"temp/{file_name}"
            
            # Download current data
            if not self.nas_client.download_file(nas_file_path, temp_file):
                logger.error(f"Failed to download job data for {job_id}")
                return False
            
            # Update data
            with open(temp_file, 'r') as f:
                job_data = json.load(f)
            
            job_data['proposal']['reviewed'] = True
            job_data['proposal']['approved'] = approved
            job_data['proposal']['reviewer_notes'] = reviewer_notes
            job_data['reviewed_at'] = datetime.now().isoformat()
            
            # Save updated data
            with open(temp_file, 'w') as f:
                json.dump(job_data, f, indent=2)
            
            # Upload back to NAS
            success = self.nas_client.upload_file(temp_file, nas_file_path)
            
            # Create approval/rejection log
            if success:
                status = "APPROVED" if approved else "REJECTED"
                log_entry = {
                    "job_id": job_id,
                    "action": status,
                    "reviewer_notes": reviewer_notes,
                    "timestamp": datetime.now().isoformat()
                }
                
                log_file = f"temp/review_log_{job_id}.json"
                with open(log_file, 'w') as f:
                    json.dump(log_entry, f, indent=2)
                
                log_path = f"{settings.nas_base_path}/freelance/logs/review_log_{job_id}.json"
                self.nas_client.upload_file(log_file, log_path)
                
                logger.info(f"Proposal {job_id} {status.lower()}")
            
            # Cleanup
            try:
                os.remove(temp_file)
                if 'log_file' in locals():
                    os.remove(log_file)
            except:
                pass
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating proposal status: {e}")
            return False
    
    def get_agent_statistics(self) -> Dict:
        """Get agent performance statistics"""
        try:
            stats = {
                "total_jobs_processed": 0,
                "proposals_generated": 0,
                "proposals_approved": 0,
                "proposals_rejected": 0,
                "pending_review": 0,
                "avg_fit_score": 0,
                "top_clients": [],
                "skill_matches": {},
                "last_processing": None
            }
            
            # Get all job files
            files = self.nas_client.list_files(f"{settings.nas_base_path}/freelance/jobs")
            
            fit_scores = []
            clients = {}
            skills = {}
            
            for file_info in files:
                if file_info.get('name', '').startswith('job_') and file_info.get('name', '').endswith('.json'):
                    temp_file = f"temp/{file_info['name']}"
                    if self.nas_client.download_file(f"{settings.nas_base_path}/freelance/jobs/{file_info['name']}", temp_file):
                        with open(temp_file, 'r') as f:
                            job_data = json.load(f)
                        
                        stats["total_jobs_processed"] += 1
                        stats["proposals_generated"] += 1
                        
                        proposal = job_data.get('proposal', {})
                        if proposal.get('reviewed'):
                            if proposal.get('approved'):
                                stats["proposals_approved"] += 1
                            else:
                                stats["proposals_rejected"] += 1
                        else:
                            stats["pending_review"] += 1
                        
                        # Collect fit scores
                        fit_score = job_data['analysis']['overall_fit_score']
                        fit_scores.append(fit_score)
                        
                        # Count clients
                        client = job_data['job']['client']
                        clients[client] = clients.get(client, 0) + 1
                        
                        # Count skills
                        for skill in job_data['analysis']['matched_skills']:
                            skills[skill] = skills.get(skill, 0) + 1
                        
                        os.remove(temp_file)
            
            # Calculate averages and top items
            if fit_scores:
                stats["avg_fit_score"] = round(sum(fit_scores) / len(fit_scores), 2)
            
            stats["top_clients"] = sorted(clients.items(), key=lambda x: x[1], reverse=True)[:5]
            stats["skill_matches"] = dict(sorted(skills.items(), key=lambda x: x[1], reverse=True)[:10])
            
            # Get last processing time from logs
            log_files = self.nas_client.list_files(f"{settings.nas_base_path}/freelance/logs")
            processing_files = [f for f in log_files if f.get('name', '').startswith('processing_results_')]
            if processing_files:
                latest_log = max(processing_files, key=lambda x: x.get('name', ''))
                stats["last_processing"] = latest_log.get('name', '').replace('processing_results_', '').replace('.json', '')
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting agent statistics: {e}")
            return {"error": str(e)}
    
    def run_full_cycle(self) -> Dict:
        """Run a complete job processing cycle"""
        logger.info("Starting full freelance agent cycle...")
        
        start_time = datetime.now()
        
        # Process job feed
        processing_results = self.process_job_feed()
        
        # Get statistics
        stats = self.get_agent_statistics()
        
        # Get pending reviews
        pending = self.get_pending_proposals()
        
        end_time = datetime.now()
        cycle_time = (end_time - start_time).total_seconds()
        
        results = {
            "cycle_start": start_time.isoformat(),
            "cycle_end": end_time.isoformat(),
            "cycle_duration_seconds": cycle_time,
            "processing_results": processing_results,
            "pending_reviews": len(pending),
            "statistics": stats,
            "next_actions": []
        }
        
        # Add recommendations for next actions
        if pending:
            results["next_actions"].append(f"Review {len(pending)} pending proposals")
        
        if processing_results.get("errors"):
            results["next_actions"].append(f"Investigate {len(processing_results['errors'])} processing errors")
        
        if not processing_results.get("processed"):
            results["next_actions"].append("Check job feed for new opportunities")
        
        logger.info(f"Full cycle completed in {cycle_time:.2f} seconds")
        return results