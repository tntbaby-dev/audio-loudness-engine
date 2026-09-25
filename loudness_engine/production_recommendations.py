def generate_recommendations(diagnoses):
    """
    Convert engineering diagnoses into production recommendations.
    """

    recommendations = []

    for diagnosis in diagnoses:

        category = diagnosis["category"]
        priority = diagnosis["priority"]

        if category == "overall_level":
            recommendations.append({
                "priority": priority,
                "category": "overall_level",
                "action": (
                    "Verify gain staging and overall mix level "
                    "before making dynamics or tonal corrections."
                ),
                "reason": diagnosis["interpretation"],
            })

        elif category == "dynamics":
            recommendations.append({
                "priority": priority,
                "category": "dynamics",
                "action": (
                    "Investigate the relationship between transient "
                    "and sustained energy before applying additional compression."
                ),
                "reason": diagnosis["interpretation"],
            })

        elif category == "loudness_range":
            recommendations.append({
                "priority": priority,
                "category": "loudness_range",
                "action": (
                    "Review section-to-section level changes "
                    "and determine whether they are intentional."
                ),
                "reason": diagnosis["interpretation"],
            })

        elif category == "true_peak":
            recommendations.append({
                "priority": priority,
                "category": "true_peak",
                "action": (
                    "Reduce peak level or apply appropriate "
                    "peak control before delivery."
                ),
                "reason": diagnosis["interpretation"],
            })

    return recommendations
