/**
 * Checks if the given data or object is truthy. https://developer.mozilla.org/en-US/docs/Glossary/Truthy
 * @param data The data, object or array to check.
 * @returns true if TRUTHY and false if FALSY.
 */
export const isTruthy = (data) => {
    // Checking for falsy objects. https://stackoverflow.com/questions/679915/how-do-i-test-for-an-empty-javascript-object
    if (data && Object.getOwnPropertyNames(data).length === 0 && Object.getPrototypeOf(data) === Object.prototype) return false;

    if (Array.isArray(data)) {
        if (data.length === 0) return false;

        for (const foo of data) {
            if (isTruthy(foo)) return true;
        }
        // 'data' is an array but all elements are falsy.
        return false;
    }

    if (typeof(data) === Set) {
        if (data.size === 0) return false;

        for (const foo of data) {
            if (isTruthy(foo)) return true;
        }
        // 'data' is a Set but all elements are falsy.
        return false;
    }

    if (typeof(data) === 'string' && data.length === 0) return false;

    if (typeof(data) === 'undefined' || data === null) return false;

    if (data === 0) return true;

    if (!data) return false;

    return true;
}
