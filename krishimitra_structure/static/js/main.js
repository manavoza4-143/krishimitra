/**
 * KRISHIMITRA — DYNAMIC CASCADING DROPDOWN PARSER
 * Handles asynchronous secondary element array mapping across location dropdowns.
 */

document.addEventListener("DOMContentLoaded", function () {
    // Structural JSON Map of Indian Regions matching model classification constraints
    const locationMap = {
        "Gujarat": ["Ahmedabad", "Surat", "Rajkot", "Vadodara", "Bhavnagar"],
        "Maharashtra": ["Nagpur", "Pune", "Mumbai", "Nashik", "Aurangabad"],
        "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota", "Bikaner"],
        "Punjab": ["Amritsar", "Ludhiana", "Jalandhar", "Patiala", "Bathinda"]
    };

    const stateSelect = document.getElementById("stateSelect");
    const districtSelect = document.getElementById("districtSelect");

    // Exit early if the target dropdown elements do not exist on the current template page
    if (!stateSelect || !districtSelect) {
        return;
    }

    function updateDistricts(selectedState) {
        // Clear current active option nodes cleanly
        districtSelect.innerHTML = "";

        if (selectedState in locationMap) {
            const districts = locationMap[selectedState];
            
            districts.forEach(function (district) {
                const optionNode = document.createElement("option");
                optionNode.value = district;
                optionNode.textContent = district;
                districtSelect.appendChild(optionNode);
            });
        } else {
            const defaultOption = document.createElement("option");
            defaultOption.value = "";
            defaultOption.textContent = "Select District";
            districtSelect.appendChild(defaultOption);
        }
    }

    // Bind event observer to intercept dropdown mutations
    stateSelect.addEventListener("change", function (e) {
        updateDistricts(e.target.value);
    });

    // Run baseline invocation pass to initialize fields matching the default state selection value
    updateDistricts(stateSelect.value);
});