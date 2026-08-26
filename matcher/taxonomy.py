"""Skill taxonomy: canonical ids, surface-form aliases, and an implication DAG.

Why this module exists
----------------------
Free-text resumes and job posts spell the same competence a dozen ways ("js",
"ES6", "JavaScript") and they routinely state a *specialisation* where the job
states a *generalisation* ("PyTorch" vs "machine learning"). A matcher that
compares raw strings therefore under-credits good candidates and is trivially
gamed by keyword stuffing.

We fix both problems with one structure:

* an alias table mapping every surface form to a canonical skill id, and
* a directed acyclic *implication* graph where ``child -> parent`` means
  "someone who can do `child` demonstrably has at least partial command of
  `parent`" (pytorch -> deep-learning -> machine-learning).

Implication credit decays geometrically with path length so that an implied
skill never outranks first-hand evidence. The reverse graph (parent -> child)
is a prerequisite graph, which is exactly what gap analysis needs to compute a
shortest upskilling path.
"""
from __future__ import annotations

import heapq
import re
from collections import deque
from dataclasses import dataclass, field

# Credit multiplier applied per implication hop. 0.72 chosen so a two-hop
# implication (pytorch -> machine-learning) still clears the 0.5 "partial
# evidence" bar while a three-hop one does not.
IMPLICATION_DECAY = 0.72

# (canonical_id, display_name, category, difficulty 1-5, extra surface forms)
SKILL_ROWS: tuple[tuple[str, str, str, int, tuple[str, ...]], ...] = (
    # --- languages -------------------------------------------------------
    ("python", "Python", "language", 2, ("py",)),
    ("javascript", "JavaScript", "language", 2, ("js", "es6", "ecmascript", "es2015")),
    ("typescript", "TypeScript", "language", 2, ("ts",)),
    ("java", "Java", "language", 3, ()),
    ("scala", "Scala", "language", 4, ()),
    # Display deliberately keeps the "(golang)" suffix: a bare "Go" alias would
    # match the English verb in every other sentence of a resume.
    ("golang", "Go (golang)", "language", 3, ("go lang", "go language")),
    ("cpp", "C++", "language", 5, ("c++", "cplusplus", "c plus plus")),
    ("sql", "SQL", "language", 2, ("structured query language",)),
    ("bash", "Bash/Shell", "language", 2, ("shell scripting", "shell", "zsh")),
    # --- web / frameworks ------------------------------------------------
    ("react", "React", "frontend", 3, ("react.js", "reactjs")),
    ("angular", "Angular", "frontend", 3, ("angularjs", "angular.js")),
    ("vue", "Vue", "frontend", 3, ("vue.js", "vuejs")),
    ("html-css", "HTML/CSS", "frontend", 1, ("html", "css", "html/css", "scss", "sass")),
    ("nodejs", "Node.js", "backend", 2, ("node.js", "node", "nodejs")),
    ("django", "Django", "backend", 3, ()),
    ("flask", "Flask", "backend", 2, ()),
    ("fastapi", "FastAPI", "backend", 2, ("fast api",)),
    ("spring", "Spring", "backend", 3, ("spring boot", "springboot")),
    # --- data ------------------------------------------------------------
    ("pandas", "pandas", "data", 2, ()),
    ("numpy", "NumPy", "data", 2, ()),
    ("spark", "Apache Spark", "data", 4, ("pyspark", "apache spark", "spark sql")),
    ("kafka", "Apache Kafka", "data", 4, ("apache kafka",)),
    ("airflow", "Airflow", "data", 3, ("apache airflow",)),
    ("dbt", "dbt", "data", 2, ("data build tool",)),
    ("etl", "ETL Pipelines", "data", 3, ("elt", "etl pipeline", "etl pipelines", "data pipelines")),
    ("data-modeling", "Data Modeling", "data", 3, ("data modelling", "dimensional modeling", "star schema")),
    ("data-engineering", "Data Engineering", "data", 4, ()),
    ("snowflake", "Snowflake", "data", 3, ()),
    ("postgresql", "PostgreSQL", "data", 3, ("postgres", "psql")),
    ("mysql", "MySQL", "data", 2, ()),
    ("mongodb", "MongoDB", "data", 3, ("mongo",)),
    ("redis", "Redis", "data", 2, ()),
    ("elasticsearch", "Elasticsearch", "data", 3, ("elastic search", "opensearch")),
    ("information-retrieval", "Information Retrieval", "ml", 4, ("search relevance", "semantic search")),
    # --- ml / ai ---------------------------------------------------------
    ("statistics", "Statistics", "ml", 3, ("statistical modeling", "statistical modelling", "stats")),
    ("machine-learning", "Machine Learning", "ml", 4, ("ml", "machine learning", "predictive modeling")),
    ("deep-learning", "Deep Learning", "ml", 4, ("dl", "deep learning", "neural networks")),
    ("pytorch", "PyTorch", "ml", 4, ("torch", "py torch")),
    ("tensorflow", "TensorFlow", "ml", 4, ("tf", "tensor flow")),
    ("keras", "Keras", "ml", 3, ()),
    ("scikit-learn", "scikit-learn", "ml", 3, ("sklearn", "scikit learn")),
    ("nlp", "NLP", "ml", 4, ("natural language processing", "text mining")),
    ("computer-vision", "Computer Vision", "ml", 4, ("image recognition", "object detection")),
    ("transformers", "Transformer Models", "ml", 4, ("transformer models", "bert", "huggingface", "hugging face")),
    ("llm", "Large Language Models", "ml", 4, ("llms", "large language models", "large language model")),
    ("rag", "RAG", "ml", 4, ("retrieval augmented generation", "retrieval-augmented generation")),
    ("recommender-systems", "Recommender Systems", "ml", 4, ("recsys", "recommendation systems", "recommender system")),
    ("time-series", "Time Series", "ml", 4, ("time-series forecasting", "time series", "forecasting")),
    ("reinforcement-learning", "Reinforcement Learning", "ml", 5, ("rl",)),
    ("feature-engineering", "Feature Engineering", "ml", 3, ("feature store",)),
    ("model-deployment", "Model Deployment", "ml", 3, ("model serving", "inference serving")),
    ("mlops", "MLOps", "ml", 4, ("ml ops", "ml platform")),
    ("experimentation", "Experimentation / A-B Testing", "ml", 3, ("a/b testing", "ab testing", "experimentation")),
    # --- infra / cloud ---------------------------------------------------
    ("cloud", "Cloud Platforms", "infra", 3, ("cloud platforms", "public cloud")),
    ("aws", "AWS", "infra", 3, ("amazon web services",)),
    ("gcp", "GCP", "infra", 3, ("google cloud", "google cloud platform")),
    ("azure", "Azure", "infra", 3, ("microsoft azure",)),
    ("docker", "Docker", "infra", 2, ("containers", "containerization")),
    ("kubernetes", "Kubernetes", "infra", 4, ("k8s", "eks", "gke")),
    ("terraform", "Terraform", "infra", 3, ("infrastructure as code", "iac")),
    ("ci-cd", "CI/CD", "infra", 2, ("ci/cd", "continuous integration", "continuous delivery", "cicd")),
    ("devops", "DevOps", "infra", 4, ("dev ops",)),
    ("linux", "Linux", "infra", 2, ("unix",)),
    ("observability", "Observability", "infra", 3, ("monitoring", "prometheus", "grafana", "opentelemetry")),
    ("security", "Security Engineering", "infra", 4, ("appsec", "application security")),
    # --- architecture / practice -----------------------------------------
    ("api-design", "API Design", "practice", 3, ("api design",)),
    ("rest-api", "REST APIs", "practice", 2, ("rest", "rest api", "restful", "rest apis")),
    ("graphql", "GraphQL", "practice", 3, ()),
    ("grpc", "gRPC", "practice", 3, ("protobuf",)),
    ("microservices", "Microservices", "practice", 4, ("microservice architecture",)),
    ("distributed-systems", "Distributed Systems", "practice", 5, ("distributed computing",)),
    ("system-design", "System Design", "practice", 4, ("systems design", "architecture design")),
    ("testing", "Automated Testing", "practice", 2, ("unit testing", "test automation", "pytest", "tdd")),
    ("agile", "Agile Delivery", "practice", 1, ("scrum", "kanban")),
    ("mentoring", "Mentoring", "practice", 2, ("coaching",)),
    ("team-leadership", "Team Leadership", "practice", 4, ("tech lead", "engineering management", "people management")),
    ("stakeholder-management", "Stakeholder Management", "practice", 3, ("stakeholder communication", "cross-functional")),
)

# child -> parent : "child implies partial command of parent"
IMPLICATIONS: tuple[tuple[str, str], ...] = (
    # tool -> host language
    ("pandas", "python"), ("numpy", "python"), ("django", "python"),
    ("flask", "python"), ("fastapi", "python"), ("pytorch", "python"),
    ("scikit-learn", "python"), ("keras", "python"),
    ("react", "javascript"), ("vue", "javascript"), ("angular", "typescript"),
    ("typescript", "javascript"), ("nodejs", "javascript"), ("spring", "java"),
    # ml hierarchy
    ("pytorch", "deep-learning"), ("tensorflow", "deep-learning"),
    ("keras", "tensorflow"), ("deep-learning", "machine-learning"),
    ("scikit-learn", "machine-learning"), ("nlp", "machine-learning"),
    ("computer-vision", "deep-learning"), ("transformers", "deep-learning"),
    ("transformers", "nlp"), ("llm", "nlp"), ("rag", "llm"),
    ("rag", "information-retrieval"), ("elasticsearch", "information-retrieval"),
    ("recommender-systems", "machine-learning"),
    ("time-series", "machine-learning"), ("time-series", "statistics"),
    ("reinforcement-learning", "machine-learning"),
    ("feature-engineering", "machine-learning"),
    ("machine-learning", "statistics"), ("experimentation", "statistics"),
    ("mlops", "machine-learning"), ("mlops", "devops"),
    ("model-deployment", "mlops"),
    # data hierarchy
    ("spark", "distributed-systems"), ("spark", "data-engineering"),
    ("kafka", "distributed-systems"), ("kafka", "data-engineering"),
    ("airflow", "data-engineering"), ("dbt", "data-engineering"),
    ("dbt", "sql"), ("etl", "data-engineering"), ("snowflake", "sql"),
    ("snowflake", "cloud"), ("postgresql", "sql"), ("mysql", "sql"),
    ("data-engineering", "data-modeling"),
    # infra hierarchy
    ("aws", "cloud"), ("gcp", "cloud"), ("azure", "cloud"),
    ("kubernetes", "docker"), ("kubernetes", "distributed-systems"),
    ("docker", "devops"), ("terraform", "devops"), ("terraform", "cloud"),
    ("ci-cd", "devops"), ("observability", "devops"), ("devops", "linux"),
    # architecture hierarchy
    ("rest-api", "api-design"), ("graphql", "api-design"), ("grpc", "api-design"),
    ("microservices", "distributed-systems"), ("microservices", "system-design"),
    ("distributed-systems", "system-design"),
    ("team-leadership", "mentoring"), ("team-leadership", "stakeholder-management"),
)


@dataclass(frozen=True)
class Skill:
    """One canonical competence."""

    skill_id: str
    display: str
    category: str
    difficulty: int
    surface_forms: tuple[str, ...]

    @property
    def learn_weeks(self) -> float:
        """Rough time-to-competence used to cost upskilling paths."""
        return float(self.difficulty) * 2.0


@dataclass
class SkillMention:
    """A surface-form hit in a document, kept with its span for explainability."""

    skill_id: str
    surface: str
    start: int
    end: int
    snippet: str = ""


@dataclass
class UpskillStep:
    """One hop of a learning plan."""

    skill_id: str
    weeks: float
    reason: str


@dataclass
class UpskillPath:
    """Shortest prerequisite-respecting route from what a person knows to a target."""

    target: str
    steps: list[UpskillStep] = field(default_factory=list)
    from_skill: str | None = None

    @property
    def weeks(self) -> float:
        return round(sum(s.weeks for s in self.steps), 1)


class SkillTaxonomy:
    """Alias resolution, text extraction, implication credit and upskilling paths."""

    def __init__(
        self,
        rows: tuple[tuple[str, str, str, int, tuple[str, ...]], ...] = SKILL_ROWS,
        implications: tuple[tuple[str, str], ...] = IMPLICATIONS,
        decay: float = IMPLICATION_DECAY,
    ) -> None:
        self.decay = decay
        self.skills: dict[str, Skill] = {}
        self._alias_to_id: dict[str, str] = {}
        for sid, display, category, difficulty, extra in rows:
            if sid in self.skills:
                raise ValueError(f"duplicate skill id {sid!r}")
            forms = {sid.replace("-", " "), sid, display.lower(), *(e.lower() for e in extra)}
            skill = Skill(sid, display, category, difficulty, tuple(sorted(forms)))
            self.skills[sid] = skill
            # sorted(): set iteration order is hash-seed dependent, and this loop
            # decides which skill owns a shared surface form. Determinism first.
            for form in sorted(forms):
                # First writer wins so a canonical id can never be stolen by an alias.
                self._alias_to_id.setdefault(form, sid)

        self.parents: dict[str, set[str]] = {s: set() for s in self.skills}
        self.children: dict[str, set[str]] = {s: set() for s in self.skills}
        for child, parent in implications:
            if child not in self.skills or parent not in self.skills:
                raise ValueError(f"implication references unknown skill: {child} -> {parent}")
            self.parents[child].add(parent)
            self.children[parent].add(child)
        self._assert_acyclic()

        self._ancestors: dict[str, dict[str, int]] = {
            s: self._bfs_depths(s, self.parents) for s in self.skills
        }
        self._pattern = self._build_pattern()

    # ------------------------------------------------------------------ graph
    def _assert_acyclic(self) -> None:
        """A cycle would make implication credit self-reinforcing and unbounded."""
        colour: dict[str, int] = {}
        for root in self.skills:
            if colour.get(root):
                continue
            stack = [(root, iter(self.parents[root]))]
            colour[root] = 1
            while stack:
                node, it = stack[-1]
                nxt = next(it, None)
                if nxt is None:
                    colour[node] = 2
                    stack.pop()
                    continue
                if colour.get(nxt) == 1:
                    raise ValueError(f"implication graph has a cycle through {nxt!r}")
                if colour.get(nxt) is None:
                    colour[nxt] = 1
                    stack.append((nxt, iter(self.parents[nxt])))

    @staticmethod
    def _bfs_depths(start: str, adj: dict[str, set[str]]) -> dict[str, int]:
        depths: dict[str, int] = {}
        queue = deque([(start, 0)])
        while queue:
            node, d = queue.popleft()
            for nxt in adj[node]:
                if nxt not in depths or depths[nxt] > d + 1:
                    depths[nxt] = d + 1
                    queue.append((nxt, d + 1))
        return depths

    def ancestors(self, skill_id: str) -> dict[str, int]:
        """All skills implied by ``skill_id`` mapped to their shortest hop count."""
        return self._ancestors.get(skill_id, {})

    def credit(self, have: str, want: str) -> float:
        """Evidence weight that holding ``have`` gives toward requirement ``want``.

        Direction matters: PyTorch implies machine learning, machine learning
        does NOT imply PyTorch, so the reverse query returns 0.0.
        """
        if have == want:
            return 1.0
        depth = self._ancestors.get(have, {}).get(want)
        return self.decay ** depth if depth else 0.0

    def best_credit(self, held: dict[str, float] | set[str] | list[str], want: str) -> tuple[float, str | None]:
        """Best (credit, source_skill) over everything a person holds."""
        best, src = 0.0, None
        for have in held:
            c = self.credit(have, want)
            if c > best:
                best, src = c, have
        return best, src

    def expand(self, skill_ids: set[str]) -> dict[str, float]:
        """Skill set plus every implied skill, each at its best credit."""
        out: dict[str, float] = {}
        for sid in skill_ids:
            out[sid] = 1.0
            for anc, depth in self._ancestors.get(sid, {}).items():
                out[anc] = max(out.get(anc, 0.0), self.decay ** depth)
        return out

    # ------------------------------------------------------------- upskilling
    def upskill_path(self, held: set[str], target: str) -> UpskillPath:
        """Cheapest prerequisite-respecting route from ``held`` to ``target``.

        Learning is the reverse of implication: to acquire PyTorch you descend
        from machine-learning to deep-learning to pytorch. Dijkstra over the
        child edges with per-skill time-to-competence as edge cost gives the
        shortest realistic plan; unreachable targets are costed from scratch
        with a 50% penalty for having no adjacent foundation.
        """
        if target not in self.skills:
            raise KeyError(target)
        if target in held:
            return UpskillPath(target=target, steps=[], from_skill=target)

        dist: dict[str, float] = {}
        prev: dict[str, str] = {}
        heap: list[tuple[float, str]] = []
        for sid in held:
            if sid in self.skills:
                dist[sid] = 0.0
                heapq.heappush(heap, (0.0, sid))
        while heap:
            d, node = heapq.heappop(heap)
            if d > dist.get(node, float("inf")):
                continue
            if node == target:
                break
            for child in sorted(self.children[node]):
                cost = d + self.skills[child].learn_weeks
                if cost < dist.get(child, float("inf")):
                    dist[child] = cost
                    prev[child] = node
                    heapq.heappush(heap, (cost, child))

        if target not in dist:
            skill = self.skills[target]
            return UpskillPath(
                target=target,
                steps=[UpskillStep(target, round(skill.learn_weeks * 1.5, 1), "no adjacent foundation; learn from scratch")],
                from_skill=None,
            )

        chain: list[str] = [target]
        while chain[-1] in prev:
            chain.append(prev[chain[-1]])
        chain.reverse()
        root = chain[0]
        steps = [
            UpskillStep(sid, self.skills[sid].learn_weeks, f"builds on {chain[i]}")
            for i, sid in enumerate(chain[1:])
        ]
        return UpskillPath(target=target, steps=steps, from_skill=root)

    # ---------------------------------------------------------- text matching
    def _build_pattern(self) -> re.Pattern[str]:
        forms: list[str] = sorted(self._alias_to_id, key=lambda f: (-len(f), f))
        # '+', '#' and '.' belong to real skill names (c++, c#, node.js) so they
        # must not act as word boundaries. '/' deliberately DOES act as one:
        # multi-skill shorthand like "JS/ES6" or "AWS/GCP" is common in resumes,
        # and longest-form-first alternation still keeps "ci/cd" intact.
        body = "|".join(re.escape(f).replace(r"\ ", r"\s+") for f in forms)
        return re.compile(r"(?<![a-z0-9+#.])(?:" + body + r")(?![a-z0-9+#])", re.IGNORECASE)

    def resolve(self, surface: str) -> str | None:
        """Map a single surface form to its canonical id (None if unknown)."""
        return self._alias_to_id.get(surface.strip().lower())

    def extract(self, text: str, context: int = 46) -> list[SkillMention]:
        """All skill mentions in ``text`` with character spans and a snippet.

        Spans are what makes a match auditable: a recruiter can see the exact
        sentence that earned the credit.
        """
        mentions: list[SkillMention] = []
        for m in self._pattern.finditer(text):
            # Aliases are stored single-spaced but the pattern tolerates line
            # wraps, so collapse whitespace before looking the surface form up.
            sid = self._alias_to_id.get(re.sub(r"\s+", " ", m.group(0).lower()))
            if sid is None:
                continue
            lo = max(0, m.start() - context)
            hi = min(len(text), m.end() + context)
            snippet = " ".join(text[lo:hi].split())
            mentions.append(SkillMention(sid, m.group(0), m.start(), m.end(), snippet))
        return mentions

    def display(self, skill_id: str) -> str:
        skill = self.skills.get(skill_id)
        return skill.display if skill else skill_id
