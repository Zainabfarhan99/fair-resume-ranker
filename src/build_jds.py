"""
build_jds.py
============
Define the synthetic Job Descriptions used for skill-match labelling.

Originally 22 JDs covering Java/web/database/cloud roles. Extended to 26 JDs
after diagnostic showed 51% of candidates matched zero JDs at 50% threshold.
The new JDs cover infrastructure, operations, and senior-DBA work that the
original developer-heavy JD set missed.

Each JD has:
    - jd_id (string, used as filename-safe identifier)
    - role_title (display name for the JD)
    - required_skills (set of canonical skill strings — must be from the
      cleaned vocabulary, i.e. skills appearing 50+ times in the full corpus)
    - description_text (natural-language JD that the ranker will see)

DESIGN PRINCIPLES (defendable to reviewers):
1. JDs describe ROLES, not CANDIDATES. Never mention career history,
   gaps, employment continuity — to keep gap-bias out of the LABEL.
2. Required skills drawn from the dataset's high-frequency vocabulary
   so real candidates can plausibly match.
3. Each JD lists 4-8 required skills (most ATS JDs work in this range).
4. Coverage spans the major role clusters in the 54K corpus including
   developer, infrastructure, database, operations, and management roles.

Run from repo root:
    python src/build_jds.py
"""

from pathlib import Path
import json

REPO_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = REPO_ROOT / "data" / "processed"
PROC_DIR.mkdir(exist_ok=True, parents=True)
OUT_PATH = PROC_DIR / "jds.json"

# -----------------------------------------------------------------------------
# 26 JDs covering the dataset's role clusters
# -----------------------------------------------------------------------------
JDS = [
    # --- existing 22 (developer/web/cloud focus) -----------------------------
    {
        "jd_id": "java_backend_dev",
        "role_title": "Java Backend Developer",
        "required_skills": {"java", "spring", "hibernate", "sql", "j2ee", "junit", "maven"},
        "description_text": (
            "Java Backend Developer\n\n"
            "We are seeking a Backend Developer with strong experience in Java enterprise "
            "development. You will design and build server-side services, integrate with "
            "relational databases, and write automated tests. The role involves working "
            "with Spring framework, Hibernate ORM, J2EE patterns, SQL, JUnit for testing, "
            "and Maven for build management. Solid object-oriented design skills required."
        ),
    },
    {
        "jd_id": "java_microservices_dev",
        "role_title": "Java Microservices Developer",
        "required_skills": {"java", "spring", "spring boot", "rest", "docker", "jenkins", "git"},
        "description_text": (
            "Java Microservices Developer\n\n"
            "Build cloud-native Java services in a microservices architecture. You will "
            "develop REST APIs using Spring Boot, containerise services with Docker, and "
            "work in continuous-delivery pipelines using Jenkins. Strong Git workflow "
            "experience expected. The team values clean API design and test-driven development."
        ),
    },
    {
        "jd_id": "frontend_dev",
        "role_title": "Frontend Developer",
        "required_skills": {"javascript", "html", "css", "html5", "css3", "jquery", "ajax", "bootstrap"},
        "description_text": (
            "Frontend Developer\n\n"
            "Build responsive, interactive web interfaces. You will work with JavaScript, "
            "HTML5, CSS3, jQuery, and Bootstrap to deliver polished user experiences. "
            "Comfort with AJAX-based dynamic content and cross-browser compatibility "
            "expected. Eye for design detail and attention to accessibility a plus."
        ),
    },
    {
        "jd_id": "react_dev",
        "role_title": "React Frontend Developer",
        "required_skills": {"javascript", "html", "css", "react", "node.js", "git", "rest"},
        "description_text": (
            "React Frontend Developer\n\n"
            "Modern frontend developer fluent in React. You will build component-based UIs "
            "consuming REST APIs, work in a Node.js tooling ecosystem, and collaborate via Git. "
            "Solid JavaScript fundamentals (HTML, CSS) required. Experience with state "
            "management libraries and frontend testing frameworks is valued."
        ),
    },
    {
        "jd_id": "angular_dev",
        "role_title": "Angular Developer",
        "required_skills": {"javascript", "html", "css", "angularjs", "rest", "git", "bootstrap"},
        "description_text": (
            "Angular Developer\n\n"
            "Frontend developer specialising in AngularJS. Build single-page applications "
            "consuming RESTful services, with responsive layouts using Bootstrap. Strong "
            "JavaScript foundations (HTML, CSS) and Git collaboration skills required."
        ),
    },
    {
        "jd_id": "fullstack_dev",
        "role_title": "Full-Stack Web Developer",
        "required_skills": {"javascript", "html", "css", "java", "spring", "sql", "git"},
        "description_text": (
            "Full-Stack Web Developer\n\n"
            "Full-stack developer comfortable across the stack. Frontend work in JavaScript, "
            "HTML, and CSS. Backend services in Java with Spring framework against SQL "
            "databases. Git-based workflow and code reviews. Comfort owning a feature "
            "end-to-end is essential."
        ),
    },
    {
        "jd_id": "python_dev",
        "role_title": "Python Backend Developer",
        "required_skills": {"python", "django", "sql", "rest", "git", "linux"},
        "description_text": (
            "Python Backend Developer\n\n"
            "Build Python web services using Django. The role requires solid SQL skills, "
            "experience designing REST APIs, comfort working in Linux environments, and "
            "Git-based collaboration. Test coverage and code quality matter to the team."
        ),
    },
    {
        "jd_id": "python_data_dev",
        "role_title": "Python Data Engineer",
        "required_skills": {"python", "sql", "linux", "aws", "git"},
        "description_text": (
            "Python Data Engineer\n\n"
            "Build data pipelines and analytical tooling in Python. Strong SQL skills "
            "essential. You will work in Linux environments, with AWS-hosted infrastructure, "
            "using Git for version control. Experience with large-scale data processing "
            "frameworks is a plus."
        ),
    },
    {
        "jd_id": "devops_engineer",
        "role_title": "DevOps Engineer",
        "required_skills": {"linux", "aws", "jenkins", "docker", "git", "bash"},
        "description_text": (
            "DevOps Engineer\n\n"
            "Own the build and deployment infrastructure. You will work with Linux, "
            "AWS cloud services, Jenkins for CI/CD, Docker for containerisation, and "
            "Bash scripting for automation. Strong Git workflow experience required. "
            "Curiosity about reliability and observability welcomed."
        ),
    },
    {
        "jd_id": "cloud_engineer",
        "role_title": "AWS Cloud Engineer",
        "required_skills": {"aws", "linux", "python", "docker", "git"},
        "description_text": (
            "AWS Cloud Engineer\n\n"
            "Design and operate AWS-based infrastructure. Linux administration skills, "
            "Python automation, Docker for service packaging, and Git for infrastructure-as-"
            "code workflows. Familiarity with multiple AWS services and security best "
            "practices expected."
        ),
    },
    {
        "jd_id": "dba_oracle",
        "role_title": "Oracle Database Administrator",
        "required_skills": {"oracle", "sql", "pl/sql", "linux", "unix"},
        "description_text": (
            "Oracle Database Administrator\n\n"
            "Administer and tune Oracle database systems. Strong SQL and PL/SQL skills "
            "required. You will work in Linux and Unix environments, manage backups and "
            "recoveries, and support production workloads. Strong attention to detail and "
            "comfort with after-hours support windows expected."
        ),
    },
    {
        "jd_id": "dba_sqlserver",
        "role_title": "SQL Server Database Administrator",
        "required_skills": {"sql server", "sql", "oracle", "active directory"},
        "description_text": (
            "SQL Server Database Administrator\n\n"
            "Administer Microsoft SQL Server environments. You will design schemas, tune "
            "queries, manage backups, and integrate with Active Directory for authentication. "
            "Familiarity with Oracle is a plus for cross-platform work."
        ),
    },
    {
        "jd_id": "data_engineer",
        "role_title": "Data Engineer",
        "required_skills": {"sql", "python", "linux", "aws", "mongodb"},
        "description_text": (
            "Data Engineer\n\n"
            "Build and maintain data infrastructure. Strong SQL fundamentals, Python "
            "scripting, and Linux comfort are required. The role involves AWS-hosted data "
            "platforms and document stores like MongoDB. Strong sense of data quality and "
            "pipeline reliability essential."
        ),
    },
    {
        "jd_id": "network_admin",
        "role_title": "Network Administrator",
        "required_skills": {"linux", "active directory", "security", "unix"},
        "description_text": (
            "Network Administrator\n\n"
            "Administer enterprise network infrastructure. Linux and Unix systems "
            "administration, Active Directory management, and a security-first mindset "
            "are required. You will support a mid-sized environment and respond to "
            "operational incidents."
        ),
    },
    {
        "jd_id": "security_analyst",
        "role_title": "IT Security Analyst",
        "required_skills": {"security", "active directory", "linux", "network security"},
        "description_text": (
            "IT Security Analyst\n\n"
            "Monitor and respond to security incidents. Required skills include security "
            "operations, Active Directory administration, Linux fundamentals, and network "
            "security concepts. Curiosity about threat landscape and incident response "
            "playbooks valued."
        ),
    },
    {
        "jd_id": "qa_engineer",
        "role_title": "QA Automation Engineer",
        "required_skills": {"java", "junit", "selenium", "git", "agile"},
        "description_text": (
            "QA Automation Engineer\n\n"
            "Design and implement automated test suites. Strong Java skills with JUnit "
            "for unit testing and Selenium for UI automation. Git collaboration in agile "
            "delivery teams. Strong analytical thinking about edge cases is essential."
        ),
    },
    {
        "jd_id": "scrum_pm",
        "role_title": "Scrum Master / Project Manager",
        "required_skills": {"agile", "jira", "project management", "scrum"},
        "description_text": (
            "Scrum Master / Project Manager\n\n"
            "Facilitate agile delivery for software teams. Strong agile/scrum facilitation "
            "skills, Jira-based backlog management, and overall project management discipline "
            "required. Confident facilitation of sprint ceremonies and stakeholder communication."
        ),
    },
    {
        "jd_id": "business_analyst",
        "role_title": "Business Analyst",
        "required_skills": {"sql", "agile", "jira", "project management"},
        "description_text": (
            "Business Analyst\n\n"
            "Bridge between business stakeholders and engineering. SQL fluency for data "
            "analysis, agile delivery experience, and Jira-based requirement management. "
            "Strong written communication and stakeholder management skills expected."
        ),
    },
    {
        "jd_id": "java_lead",
        "role_title": "Senior Java Developer / Tech Lead",
        "required_skills": {"java", "spring", "hibernate", "sql", "j2ee", "agile", "junit", "maven"},
        "description_text": (
            "Senior Java Developer / Tech Lead\n\n"
            "Lead a team of Java developers building enterprise applications. Deep "
            "expertise in Java, Spring, Hibernate, J2EE patterns, SQL, JUnit, and Maven. "
            "Comfort facilitating agile ceremonies and mentoring junior engineers. "
            "Architectural design experience valued."
        ),
    },
    {
        "jd_id": "node_dev",
        "role_title": "Node.js Backend Developer",
        "required_skills": {"node.js", "javascript", "rest", "mongodb", "git"},
        "description_text": (
            "Node.js Backend Developer\n\n"
            "Build server-side services in Node.js. Strong JavaScript fundamentals, REST "
            "API design, MongoDB document modelling, and Git collaboration required. "
            "Comfort with asynchronous programming patterns essential."
        ),
    },
    {
        "jd_id": "php_dev",
        "role_title": "PHP Web Developer",
        "required_skills": {"php", "javascript", "html", "css", "mysql", "jquery"},
        "description_text": (
            "PHP Web Developer\n\n"
            "Build web applications in PHP. Strong frontend complement (JavaScript, HTML, "
            "CSS, jQuery) and MySQL database experience required. Comfort working in "
            "established codebases and incrementally improving them."
        ),
    },
    {
        "jd_id": "dotnet_dev",
        "role_title": ".NET Developer",
        "required_skills": {"sql", "sql server", "javascript", "html", "css"},
        "description_text": (
            ".NET Developer\n\n"
            "Build enterprise applications on the .NET stack. Strong SQL Server and SQL "
            "fundamentals, with web frontend skills (JavaScript, HTML, CSS) for full-stack "
            "feature delivery. Familiarity with Microsoft ecosystem expected."
        ),
    },

    # --- new 4 JDs covering the missing clusters -----------------------------
    {
        "jd_id": "senior_dba_ops",
        "role_title": "Senior Database Engineer (Operations)",
        "required_skills": {"oracle", "rman", "data guard", "performance tuning", "backup and recovery", "disaster recovery"},
        "description_text": (
            "Senior Database Engineer\n\n"
            "Senior Oracle database engineer focused on production database operations. "
            "Deep experience with RMAN backups and recoveries, Data Guard for replication, "
            "performance tuning of production workloads, and disaster recovery planning. "
            "You will own the operational health of mission-critical Oracle systems and "
            "lead recovery exercises."
        ),
    },
    {
        "jd_id": "network_engineer",
        "role_title": "Network Engineer (Cisco / VMware)",
        "required_skills": {"cisco", "dns", "dhcp", "vpn", "firewalls", "vmware"},
        "description_text": (
            "Network Engineer\n\n"
            "Operate and maintain enterprise network infrastructure. The role spans Cisco "
            "switching and routing, DNS and DHCP administration, VPN configuration, "
            "firewall management, and VMware-based virtual networking. Strong "
            "troubleshooting instincts and comfort during incident response are essential."
        ),
    },
    {
        "jd_id": "systems_admin",
        "role_title": "Systems Administrator (Windows / Infrastructure)",
        "required_skills": {"windows", "active directory", "vmware", "sharepoint", "dns"},
        "description_text": (
            "Systems Administrator\n\n"
            "Administer enterprise Windows-based infrastructure. The role includes Active "
            "Directory administration, VMware virtualisation management, SharePoint "
            "support, and DNS services. You will support end-users, manage server lifecycle, "
            "and coordinate with networking and security teams."
        ),
    },
    {
        "jd_id": "database_developer",
        "role_title": "Database Developer (Multi-Platform)",
        "required_skills": {"sql", "pl/sql", "oracle", "sql server", "performance tuning", "database design"},
        "description_text": (
            "Database Developer\n\n"
            "Design and build database solutions across Oracle and SQL Server platforms. "
            "Strong SQL and PL/SQL skills required. The role involves database design, "
            "performance tuning, and collaborating with application teams on data-access "
            "patterns. You will own schema evolution and query optimisation."
        ),
    },
]

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
def validate(jds):
    seen_ids = set()
    for jd in jds:
        assert jd["jd_id"] not in seen_ids, f"duplicate jd_id: {jd['jd_id']}"
        seen_ids.add(jd["jd_id"])
        for f in ("jd_id", "role_title", "required_skills", "description_text"):
            assert f in jd, f"missing field {f} in {jd['jd_id']}"
        for s in jd["required_skills"]:
            assert s == s.lower().strip(), f"skill not lowercase/stripped: '{s}' in {jd['jd_id']}"
        n = len(jd["required_skills"])
        assert 4 <= n <= 9, f"jd {jd['jd_id']} has {n} skills (expected 4-8)"

validate(JDS)

# -----------------------------------------------------------------------------
# Write JSON
# -----------------------------------------------------------------------------
serialisable = []
for jd in JDS:
    serialisable.append({
        "jd_id": jd["jd_id"],
        "role_title": jd["role_title"],
        "required_skills": sorted(list(jd["required_skills"])),
        "description_text": jd["description_text"],
    })

with open(OUT_PATH, "w") as f:
    json.dump(serialisable, f, indent=2)

print(f"validated and wrote {len(serialisable)} JDs to {OUT_PATH}")
print()
print("JD overview:")
for jd in serialisable:
    print(f"  {jd['jd_id']:25s}  {jd['role_title']:50s}  ({len(jd['required_skills'])} skills)")