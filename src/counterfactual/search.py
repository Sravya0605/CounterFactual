"""Counterfactual search engine skeleton."""
import logging
import time
from typing import Any, Dict, List, Optional

import networkx as nx

from src.counterfactual import feasibility, substitutions

try:
    from src.utils.pyg_adapter import build_api_vocab, graph_to_pyg_data
    import torch
except Exception:  # pragma: no cover - optional dependency
    build_api_vocab = None
    graph_to_pyg_data = None
    torch = None

logger = logging.getLogger(__name__)


class CounterfactualSearch:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.max_edits = 10
        self.max_candidates = 2000
        # Separate insertion candidate budget so deletions/substitutions
        # cannot starve insert proposals on large graphs.
        self.max_insertion_candidates = 50
        self.threshold = 0.5
        self.api_vocab = None
        self.enforce_feasibility = True
        self.pair_node_limit = 32
        self.max_pair_candidates = 512

        # Beyond-pair combination search.
        #
        # These are deliberately bounded because exhaustive combinations become
        # combinatorial very quickly on real malware traces.
        self.chain_node_limit = 32
        self.max_chain_starts = 8
        self.max_chain_candidates = 512
        self.chain_beam_width = 8
        self.max_edits = max(3, min(10, int(max(3, self.graph.number_of_nodes() * 0.1 + 0.999))))

    def _node_priority(self, node: str) -> tuple:
        data = self.graph.nodes[node]
        api = str(data.get("api") or "").lower()
        entity_type = str(data.get("entity_type") or "")
        security_tokens = (
            "inject", "hollow", "createremotethread", "createservice",
            "scheduledtask", "regset", "regcreate", "writefile",
            "connect", "socket", "protectvirtualmemory", "mapview",
            "createprocess", "terminateprocess", "download", "encrypt",
        )
        semantic = int(bool(data.get("attack_id"))) * 100
        semantic += sum(token in api for token in security_tokens) * 10
        semantic += int(entity_type in {"persistence", "network", "registry"}) * 8
        semantic += min(len(data.get("resources", []) or []), 4)
        routine_penalty = int(api in {"heapcreate", "getmodulehandlea", "getmodulehandlew", "ldrgetprocedureaddressforcaller"}) * 6

        # Boost by what the TRAINED CLASSIFIER actually weighs, when known --
        # not just the hand-picked security tokens above. Without this, a
        # node contributing to whatever feature the model actually keys on
        # (which can be something unglamorous like a DLL-load count) never
        # gets prioritized, and the search spends its edit budget on
        # "security-looking" nodes the model may not care about at all.
        importance_boost = 0
        feature_importance = getattr(self, "feature_importance", None)
        max_importance = getattr(self, "_max_feature_importance", None)
        if feature_importance and max_importance:
            raw = feature_importance.get(api, 0) + feature_importance.get(f"log1p_{api}", 0)
            if raw > 0:
                importance_boost = round((raw / max_importance) * 50)

        return (semantic + importance_boost - routine_penalty, -len(api), str(node))

    def _candidate_nodes(self) -> List[str]:
        # Exclude the process ANCHOR node (identified by its literal api
        # value, set once per process by the graph builder) and synthetic
        # resource-lifetime nodes. Do NOT exclude by entity_type=="process":
        # real event nodes (e.g. OpenProcess, CreateRemoteThread) are also
        # classified with entity_type "process" by _entity_type_for_api,
        # since that label describes the API *category*, not "this is an
        # anchor". Filtering on entity_type here previously made every
        # process-injection call permanently invisible to the search.
        nodes = [
            node for node, data in self.graph.nodes(data=True)
            if data.get("entity_type") != "resource"
            and str(data.get("api") or "").lower() != "process"
        ]
        return sorted(nodes, key=self._node_priority, reverse=True)

    def _apply_candidate(self, candidate: Dict) -> nx.DiGraph:
        return feasibility.apply_candidate(self.graph, candidate)

    def _candidate_cost(self, candidate: Dict) -> int:
        return feasibility.candidate_cost(candidate)

    def _within_edit_budget(self, candidate: Dict) -> bool:
        return self._candidate_cost(candidate) <= self.max_edits

    def _downstream_cascade(self, node: str) -> List[str]:
        """Return the complete resource/temporal dependency closure for a node."""
        seen = set()
        ordered = []
        stack = [node]
        while stack:
            current = stack.pop()
            successors = sorted(self.graph.successors(current), key=str, reverse=True)
            for succ in successors:
                edge_data = self.graph.get_edge_data(current, succ) or {}
                if edge_data.get("type") not in {"resource", "temporal", "process_creation"}:
                    continue
                if succ in seen:
                    continue
                seen.add(succ)
                ordered.append(succ)
                stack.append(succ)
        return ordered

    def _closure_candidate(self, node: str) -> Optional[Dict]:
        closure = self._downstream_cascade(node)
        candidate = {"delete_nodes": [node, *closure], "substitute": {}}
        return candidate if self._within_edit_budget(candidate) else None

    def _merged_closure_candidate(self, nodes: List[str]) -> Optional[Dict]:
        delete_nodes = set(nodes)
        for node in nodes:
            delete_nodes.update(self._downstream_cascade(node))
        candidate = {"delete_nodes": sorted(delete_nodes), "substitute": {}}
        return candidate if self._within_edit_budget(candidate) else None

    def _candidate_priority(self, candidate: Dict) -> float:
        """Score candidate by semantic priority of affected nodes (higher = better)."""
        score = 0.0
        for n in candidate.get("delete_nodes", []) or []:
            if n in self.graph:
                score += self._node_priority(n)[0]
        for n in (candidate.get("substitute", {}) or {}).keys():
            if n in self.graph:
                score += self._node_priority(n)[0]
        return score

    @staticmethod
    def _candidate_key(candidate: Dict) -> str:
        return repr(
            (
                tuple(sorted(candidate.get("delete_nodes", []) or [])),
                tuple(sorted(tuple(edge) for edge in candidate.get("delete_edges", []) or [])),
                tuple(sorted((candidate.get("substitute", {}) or {}).items())),
                tuple(sorted(repr(item) for item in candidate.get("insert_nodes", []) or [])),
            )
        )

    def propose(self, guidance_model: Any = None, api_vocab: Any = None, feature_importance: Any = None) -> List[Dict]:
        """Generate structurally possible edits without querying a classifier."""
        cands: List[Dict] = []
        if api_vocab is not None:
            self.api_vocab = api_vocab
        if feature_importance:
            self.feature_importance = feature_importance
            self._max_feature_importance = max(feature_importance.values()) if feature_importance else None
        nodes = self._candidate_nodes()
        seen = set()

        def add(candidate: Optional[Dict]) -> None:
            if candidate is None or not self._within_edit_budget(candidate):
                return
            key = self._candidate_key(candidate)
            if key not in seen:
                seen.add(key)
                cands.append(candidate)

        if guidance_model is not None and hasattr(guidance_model, "parameters"):
            try:
                from src.counterfactual.gradient_proposer import propose_from_gradients
                from src.counterfactual.edge_mask_proposer import propose_edge_deletions

                guided = propose_from_gradients(
                    guidance_model, self.graph, top_k=50, api_vocab=self.api_vocab
                )
                guided.extend(
                    propose_edge_deletions(
                        guidance_model, self.graph, top_k=25, api_vocab=self.api_vocab
                    )
                )
                for candidate in guided:
                    add(candidate)
                    for node in candidate.get("delete_nodes", []):
                        add(self._closure_candidate(node))
            except Exception as exc:
                logger.warning("Gradient-guided proposals unavailable: %s", exc)

        # Cost-1 node deletions are proposed first, but semantic ranking keeps
        # the bounded search focused on behavior rather than runtime noise.
        for n in nodes:
            add({"delete_nodes": [n], "substitute": {}})

        # Edge deletion is part of the edit vocabulary. Process edges are
        # excluded because removing one destroys execution context rather than
        # changing a behavior; feasibility still rejects broken dependencies.
        for u, v, data in self.graph.edges(data=True):
            if data.get("type") in {"temporal", "resource"}:
                add({"delete_nodes": [], "delete_edges": [(u, v)], "substitute": {}})

        # Add complete dependency closures, then substitutions and their
        # closures. A partial cascade is not a valid representation of the
        # deletion described by the abstract.
        for n in nodes:
            add(self._closure_candidate(n))
            api = self.graph.nodes[n].get("api", "")
            for sub in substitutions.get_substitutes(api):
                add({"delete_nodes": [], "substitute": {n: sub}})

        # Independent combinations are essential when no single behavior is
        # sufficient to cross the classifier boundary. Keep this bounded and
        # semantically ranked so large traces remain tractable.
        pair_count = 0
        for index, first in enumerate(nodes[: self.pair_node_limit]):
            for second in nodes[index + 1 : self.pair_node_limit]:
                add(self._merged_closure_candidate([first, second]))
                pair_count += 1
                if pair_count >= self.max_pair_candidates:
                    break
            if pair_count >= self.max_pair_candidates:
                break

        # Beyond pairs: search bounded combinations of independently
        # selected nodes. A single behavior or pair may not be sufficient
        # to cross the classifier boundary when malicious evidence is
        # distributed across several events.
        #
        # We deliberately do NOT enumerate all combinations because
        # C(N, K) becomes enormous on real traces. Instead, maintain a
        # bounded beam of promising partial combinations and grow them
        # incrementally.
        chain_pool = nodes[: self.chain_node_limit]

        chain_candidates_added = 0

        if len(chain_pool) >= 3 and self.max_edits >= 3:
            # Each beam entry is a tuple containing the selected node names.
            #
            # Start from several different high-priority nodes rather than
            # only the first node. This prevents the search from becoming
            # locked onto one semantic family.
            beams = [
                (node,)
                for node in chain_pool[: self.max_chain_starts]
            ]

            seen_combinations = set()

            for depth in range(2, self.max_edits + 1):
                next_beam = []

                for selected in beams:
                    selected_set = set(selected)

                    # Candidates are considered in classifier-aware /
                    # semantic-priority order because `nodes` is already
                    # sorted by _node_priority().
                    for node in chain_pool:
                        if node in selected_set:
                            continue

                        combination = tuple(
                            sorted((*selected, node), key=str)
                        )

                        if combination in seen_combinations:
                            continue

                        seen_combinations.add(combination)

                        candidate = self._merged_closure_candidate(
                            list(combination)
                        )

                        if candidate is None:
                            # Adding nodes can only increase the deletion
                            # footprint, so this combination cannot be used.
                            continue

                        add(candidate)
                        chain_candidates_added += 1

                        if chain_candidates_added >= self.max_chain_candidates:
                            break

                        # Keep this partial combination available for the
                        # next depth.
                        next_beam.append(combination)

                    if chain_candidates_added >= self.max_chain_candidates:
                        break

                if chain_candidates_added >= self.max_chain_candidates:
                    break

                if not next_beam:
                    break

                # Rank partial combinations by the semantic priority of
                # their nodes. This keeps the beam focused on combinations
                # that are more likely to affect the classifier.
                def combination_priority(combination):
                    return sum(
                        self._node_priority(node)[0]
                        for node in combination
                    )

                next_beam.sort(
                    key=lambda combo: (
                        -combination_priority(combo),
                        combo,
                    )
                )

                # Prevent exponential growth of the beam.
                beams = next_beam[: self.chain_beam_width]

        # Pass 3: insertion proposals (separate budget)
        try:
            from src.counterfactual.insertions import propose_insertions

            insert_cands = propose_insertions(self.graph, top_k=self.max_insertion_candidates, api_vocab=self.api_vocab)
            # Do not let insertions consume the main candidate budget; return
            # them appended so callers can still see a mixture.
            for candidate in insert_cands[: self.max_insertion_candidates]:
                add(candidate)
        except Exception as exc:
            logger.warning("Insertion proposer failed: %s", exc)

        return sorted(
            cands,
            key=lambda candidate: (
                self._candidate_cost(candidate),
                -self._candidate_priority(candidate),
                self._candidate_key(candidate),
            ),
        )[: self.max_candidates]

    def validate(self, candidate: Dict) -> bool:
        if not self.enforce_feasibility:
            return True
        return feasibility.validate_candidate(self.graph, candidate)

    def generate_candidates(self) -> Dict:
        """Generate and validate candidates without calling a classifier."""
        started = time.perf_counter()
        trace = []
        candidates = self.propose()
        feasible_candidates = []
        for cand in candidates:
            valid = self.validate(cand)
            trace.append({"candidate": cand, "valid": valid})
            if not valid:
                continue
            feasible_candidates.append(cand)
        result = {
            "status": "candidates_available" if feasible_candidates else "no_feasible_candidate",
            "candidate": None,
            "edited_graph": None,
            "feasible_candidates": feasible_candidates,
            "constraint_comparison": {
                "proposed": len(candidates),
                "feasible": len(feasible_candidates),
                "infeasible": len(candidates) - len(feasible_candidates),
            },
        }
        result.update(self._run_metadata(trace, started, len(candidates)))
        return result

    def generate(self) -> Dict:
        """Backward-compatible name for classifier-free candidate generation."""
        return self.generate_candidates()

    def find_flip(self, classifier: Any) -> Dict:
        """Confirm the lowest-cost feasible candidate that flips a verdict.

        Scores the ORIGINAL graph first -- one cheap classifier call. Only if
        the sample is actually above the malicious threshold do we pay for
        the expensive propose+validate search. Checking this first avoids
        wasting the entire search (seconds to tens of seconds on a large
        real graph) on a sample that was never going to need a
        counterfactual in the first place.
        """
        try:
            original_probability = float(classifier.predict_proba([self.graph])[0])
        except Exception as exc:
            return {
                "status": "unscoreable",
                "candidate": None,
                "edited_graph": None,
                "classifier_error": str(exc),
            }

        if original_probability < self.threshold:
            return {
                "status": "not_malicious",
                "candidate": None,
                "edited_graph": None,
                "orig_prob": original_probability,
                "scored_candidates": 0,
            }

        guidance_model = getattr(classifier, "model", classifier)
        generated = self._generate_with_guidance(classifier, guidance_model)
        generated["orig_prob"] = original_probability
        candidates = generated["feasible_candidates"]
        if not candidates:
            return generated

        scored = []
        for candidate in candidates:
            edited = self._apply_candidate(candidate)
            try:
                new_probability = float(classifier.predict_proba([edited])[0])
            except Exception as exc:
                logger.warning("Unable to score candidate: %s", exc)
                continue
            if new_probability < self.threshold:
                generated.update({
                    "status": "completed",
                    "candidate": candidate,
                    "edited_graph": edited,
                    "new_prob": new_probability,
                    "scored_candidates": len(scored) + 1,
                })
                return generated
            scored.append((candidate, edited, new_probability))

        generated.update({"status": "no_flip_found", "scored_candidates": len(scored)})
        return generated

    @staticmethod
    def _extract_feature_importance(classifier: Any) -> Optional[Dict[str, float]]:
        """Best-effort extraction of per-feature importance from a trained
        classifier, so node priority can reflect what the model actually
        weighs -- not just the hand-picked security-relevant tokens in
        _node_priority. Works for LightGBM (including sklearn's
        CalibratedClassifierCV wrapper); returns None for anything else,
        which leaves the existing heuristic-only ranking unchanged.
        """
        vocab = getattr(classifier, "feature_vocab", None)
        model = getattr(classifier, "model", None)
        if not vocab or model is None:
            return None

        estimator = model
        if hasattr(estimator, "calibrated_classifiers_"):
            try:
                estimator = estimator.calibrated_classifiers_[0].estimator
            except Exception:
                return None

        importances = None
        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
        elif hasattr(estimator, "booster_") and hasattr(estimator.booster_, "feature_importance"):
            importances = estimator.booster_.feature_importance(importance_type="gain")
        elif hasattr(estimator, "feature_importance"):
            try:
                importances = estimator.feature_importance(importance_type="gain")
            except Exception:
                importances = None

        if importances is None or len(importances) != len(vocab):
            return None
        return dict(zip(vocab, importances))

    def _generate_with_guidance(self, classifier: Any, guidance_model: Any) -> Dict:
        started = time.perf_counter()
        trace = []
        api_vocab = getattr(classifier, "api_vocab", None)
        feature_importance = self._extract_feature_importance(classifier)
        candidates = self.propose(guidance_model=guidance_model, api_vocab=api_vocab, feature_importance=feature_importance)
        feasible_candidates = []
        for candidate in candidates:
            valid = self.validate(candidate)
            trace.append({"candidate": candidate, "valid": valid})
            if valid:
                feasible_candidates.append(candidate)
        result = {
            "status": "candidates_available" if feasible_candidates else "no_feasible_candidate",
            "candidate": None,
            "edited_graph": None,
            "feasible_candidates": feasible_candidates,
        }
        result.update(self._run_metadata(trace, started, len(candidates)))
        return result

    @staticmethod
    def _run_metadata(trace: List[Dict], started: float, proposed: int) -> Dict:
        return {
            "search": {
                "proposed_candidates": proposed,
                "evaluated_candidates": len(trace),
                "runtime_seconds": round(time.perf_counter() - started, 6),
            },
            "candidate_trace": trace,
        }

    def _outcome(self, status: str, candidate: Optional[Dict], edited_graph: Optional[nx.DiGraph], trace: List[Dict], started: float, proposed: int) -> Dict:
        result = {
            "status": status,
            "candidate": candidate,
            "edited_graph": edited_graph,
        }
        result.update(self._run_metadata(trace, started, proposed))
        return result
